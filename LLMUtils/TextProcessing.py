from LLMUtils.ReadData import ReadFile
import re
from langchain_core.documents import Document
from LLMUtils.PrepareChunks import TextChunks
from LLMUtils.VectoreStore import Vectors
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import cleantext
import os
from dotenv import load_dotenv
import time

load_dotenv()
index_name = os.getenv("INDEX_NAME")
pinecone_api = os.getenv("PINECONE_API")

# Class for managing the text preprocessing 
# pipeline and embedding storage in Pinecone.
class PrepareText:
    """
    Reads, cleans, chunks, and vectorizes MULTIPLE PDF files
    with section detection and metadata.
    """

    def __init__(self, file_paths, config=None, api_key: str = None):
        try:
            if isinstance(file_paths, str):
                file_paths = [file_paths]

            self.file_paths = file_paths
            self.config = config
            self.api_key = api_key

            self.all_pages = []

            for file_path in self.file_paths:
                pages = ReadFile.read_pdf_pages(file_path=file_path)
                for page in pages:
                    page["source"] = file_path
                self.all_pages.extend(pages)

            print(f"Total pages loaded from all PDFs: {len(self.all_pages)}")

        except Exception as e:
            print(f"Error initializing PrepareText: {e}")
            self.all_pages = []

    def clean_data(self, text):
        try:
            return cleantext.clean(
                text,
                lowercase=False,
                punct=False,
                extra_spaces=True
            )
        except Exception:
            return text

    def detect_section(self, text):
        """
        Extracts all possible section headings from a page.
        Returns list of detected headings in order.
        """
        try:
            section_patterns = [
                r'^\d+(\.\d+)+\s+[A-Z][A-Z0-9\s\-\(\)&]+$',
                r'^\d+\.?\s+[A-Z][A-Z0-9\s\-\(\)&]+$',
                r'^Clause\s+\d+.*',
                r'^[A-Z][A-Z0-9\s\-\(\)&]{5,}$'
            ]

            sections = []
            lines = text.split("\n")

            for line in lines:
                line = line.strip()

                for pattern in section_patterns:
                    if re.match(pattern, line):
                        sections.append(line)
                        break

            return sections if sections else "EMPTY_STRING"

        except Exception as e:
            print(f"Error detecting section: {e}")
            return None

    def get_chunks(self, chunk_size=1200, overlap=200, separator=None):

        splitter = TextChunks.initialize(
            separator=separator,
            chunksize=chunk_size,
            overlap=overlap
        )

        final_docs = []

        for page in self.all_pages:

            original_text = page["text"]
            cleaned_text = self.clean_data(original_text)
            detected_sections = self.detect_section(original_text)

            splits = splitter.split_text(cleaned_text)

            for j, chunk in enumerate(splits):

                metadata = {
                    "page": page["page_number"],
                    "source": page["source"],
                    "file_name": os.path.basename(page["source"]),
                    "section": detected_sections,
                    "chunk_id": f"{page['source']}_p{page['page_number']}_c{j}"
                }

                final_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                )

        print(f"Created {len(final_docs)} chunks from multiple PDFs.")
        return final_docs

    def create_text_vectors(self, separator=None, chunksize=1000, overlap=100, id: int = None, batch: int = None):

        Vectors.initialize(config=self.config)

        chunks = self.get_chunks(
            chunk_size=chunksize,
            overlap=overlap,
            separator=separator
        )
        
        for doc in chunks:print(doc.metadata)
        vectors = Vectors.generate_vectors_from_documents(chunks=chunks,user_id=id,batch_size=batch)

        if vectors:
            print("Vector store successfully created for multiple PDFs.")
        else:
            print("Vector store creation failed.")

        return vectors


# Responsible for ingesting data into and retrieving data from the Pinecone index.
# Applies file-level filtering based on user queries.
class PineconeManager:


    def __init__(self, config=None):
        try:
            self.pc = Pinecone(api_key=pinecone_api)
            self.index_name = index_name
            self.index = self.pc.Index(index_name)
            self.config=config
            self.embedding_model = Vectors.initialize(config=self.config)
        except Exception as e:
            print(f"[Pinecone] Init error: {e}")

    
    # Method to check if embeddings alreasy exists for a file (Supports only single file)
    def embeddings_exist(self, user_id: str, file_name: str):

        try:
            dummy_vector = self.embedding_model.embed_query("check")

            response = self.index.query(
                vector=dummy_vector,
                top_k=1,
                namespace="__default__",
                filter={
                    "user_id": user_id,
                    "file_name": file_name
                },
                include_metadata=True
            )

            return len(response["matches"]) > 0

        except Exception as e:
            print(f"[Pinecone] Error checking embeddings: {e}")
            return False


    # Method to check if embeddings alreasy exists for a file (Supports Multiple files)
    def embeddings_exist_multi(self, user_id: str, file_names: list):

        try:
            existing_files = list()
            missing_files = list()

            for file_name in file_names:
                if self.embeddings_exist(user_id, file_name):
                    existing_files.append(file_name)
                else:
                    missing_files.append(file_name)

            return existing_files, missing_files

        except Exception as e:
            print(f"[Pinecone] Error in multi check: {e}")
            return [], file_names

    # Method to load the Vector
    def load_vector_store(self):

        try:
            return PineconeVectorStore(
                index=self.index,
                embedding=self.embedding_model,
                namespace="__default__"
            )
        except Exception as e:
            print(f"[Pinecone] Error loading vector store: {e}")
            return None


    # Method to get embeddings based on user_id and file_name (if already exists)
    def _build_retriever(self, user_id: str, file_names: list, k: int):

        try:
            vector_store = self.load_vector_store()
            if not vector_store:
                return None

            return vector_store.as_retriever(
                search_kwargs={
                    "k": k,
                    "filter": {
                        "user_id": user_id,
                        "file_name": {"$in": file_names}
                    }
                }
            )

        except Exception as e:
            print(f"[Pinecone] Error building retriever: {e}")
            return None


    def _format_response(self, status, retriever=None, existing=None, missing=None):

        return {
            "status": status,
            "retriever": retriever,
            "existing_files": existing or [],
            "missing_files": missing or []
        }


    # Method to Extract the File names from the Query (File level Isolation for a User)
    def extract_files_from_query(self, query: str, available_files: list):

        try:
            if not query:
                return available_files

            query = query.lower()
            matched_files = []

            for file in available_files:
                name = file.lower()
                if name in query:
                    matched_files.append(file)

            return matched_files if matched_files else available_files

        except Exception as e:
            print(f"[Pinecone] Error extracting files from query: {e}")
            return available_files


# Class to manage entire Retreival and Ingestion Pipeline
# Performs Retrieval if the User and the File already present else switch to Ingestion.
class RetrieverService:

    def __init__(self, file_paths: list, user_id: int, config,gemini_api: str):
        
        self.file_paths = file_paths
        self.user_id = str(user_id)
        self.config = config
        self.file_names = self._extract_file_names()
        self.api_key = gemini_api
        self.pm = PineconeManager(
            config=config
        )


    # funtion to extract file names
    def _extract_file_names(self):
        
        return [os.path.basename(path) for path in self.file_paths]


    # function to get missing files-
    def _get_missing_paths(self, missing_files):
        
        return [
            path for path in self.file_paths
            if os.path.basename(path) in missing_files
        ]


    # Method to Ingest the file and the user id to the Pinecone Vectorstore if not already present.
    def _ingest_files(self, missing_paths: list, chunk: int, overlap: int, sep: list, batch_size: int):
        try:
            if not missing_paths:
                return

            text_processor = PrepareText(
                file_paths=missing_paths,
                config=self.config,
                api_key=self.api_key
            )

            text_processor.create_text_vectors(
                chunksize=chunk,
                overlap=overlap,
                separator=sep,
                id=self.user_id,
                batch=batch_size
            )

            print(f"Ingested files: {missing_paths}")

        except Exception as e:
            print(f"[RetrieverService] Ingestion error: {e}")

    
    # Method to keep track of the missing and existing files.
    # The files ingestion is executes first everytime the user adds a new file.
    def prepare_data(self, chunk: int, overlap: int, sep: list, batch_size: int):
        try:
            existing_files, missing_files = self.pm.embeddings_exist_multi(
            user_id=self.user_id,
            file_names=self.file_names
            )

            print(f"Existing files: {existing_files}")
            print(f"Missing files: {missing_files}")

            if missing_files:
                missing_paths = self._get_missing_paths(missing_files)

                self._ingest_files(
                    missing_paths=missing_paths,
                    chunk=chunk,
                    overlap=overlap,
                    sep=sep,
                    batch_size=batch_size
                )

                time.sleep(30)
        except Exception as e:
            return e
        
    # Filters the files based on user's query and returns the Retriever
    def get_retriever(self, query: str):
        try:
            
            filtered_files = self.pm.extract_files_from_query(
                query=query,
                available_files=self.file_names
            )

            print(f"Filtered files from query: {filtered_files}")

            
            retriever = self.pm._build_retriever(
                user_id=self.user_id,
                file_names=filtered_files,
                k=100
            )

            return retriever

        except Exception as e:
            print(f"[RetrieverService] Error: {e}")
            return None
        
        
if __name__ == "__main__":
    pass