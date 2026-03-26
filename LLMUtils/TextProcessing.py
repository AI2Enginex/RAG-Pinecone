from LLMUtils.ReadData import ReadFile
from textwrap import dedent
import re
from langchain_core.documents import Document
from LLMUtils.PrepareChunks import TextChunks
from LLMUtils.VectoreStore import Vectors
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import cleantext
import os

from dotenv import load_dotenv

load_dotenv()
pinecone_api = os.getenv("PINECONE_API")

# ========================== PREPARE TEXT ============================

class PrepareText:
    """
    Reads, cleans, chunks, and vectorizes MULTIPLE PDF files
    with section detection metadata.
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


class PineconeManager:


    def __init__(self, index_name,config=None):
        try:
            self.pc = Pinecone(api_key=pinecone_api)
            self.index_name = index_name
            self.index = self.pc.Index(index_name)
            self.config=config
            self.embedding_model = Vectors.initialize(config=self.config)
        except Exception as e:
            print(f"[Pinecone] Init error: {e}")


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


    # -----------------------------------
    # MULTI FILE CHECK
    # -----------------------------------
    def embeddings_exist_multi(self, user_id: str, file_names: list):

        try:
            existing_files = []
            missing_files = []

            for file_name in file_names:
                if self.embeddings_exist(user_id, file_name):
                    existing_files.append(file_name)
                else:
                    missing_files.append(file_name)

            return existing_files, missing_files

        except Exception as e:
            print(f"[Pinecone] Error in multi check: {e}")
            return [], file_names


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


    # -----------------------------------
    # PRIVATE: BUILD RETRIEVER
    # -----------------------------------
    def _build_retriever(self, user_id: str, file_names: list, k=40):

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


    # -----------------------------------
    # PRIVATE: FORMAT RESPONSE
    # -----------------------------------
    def _format_response(self, status, retriever=None, existing=None, missing=None):

        return {
            "status": status,
            "retriever": retriever,
            "existing_files": existing or [],
            "missing_files": missing or []
        }


    # -----------------------------------
    # Get retriever (single file - unchanged)
    # -----------------------------------
    def get_retriever(self, user_id: str, file_name: str, k=40):

        try:
            if self.embeddings_exist(user_id=user_id, file_name=file_name) is False:
                return "Invalid Parameters"
            
            return self.load_vector_store().as_retriever(
                search_kwargs={
                    "k": k,
                    "filter": {
                        "user_id": user_id,
                        "file_name": file_name
                    }
                }
            )

        except Exception as e:
            print(f"[Pinecone] Error getting retriever: {e}")
            return None


    # -----------------------------------
    # CLEAN MULTI FILE RETRIEVER
    # -----------------------------------
    def get_retriever_multi(self, user_id: str, file_names: list, k=40):

        try:
            existing_files, missing_files = self.embeddings_exist_multi(user_id, file_names)

            print(f"Existing files: {existing_files}")
            print(f"Missing files: {missing_files}")

            # determine status
            if not existing_files:
                return self._format_response(
                    status="missing_all",
                    missing=missing_files
                )

            retriever = self._build_retriever(
                user_id=user_id,
                file_names=existing_files if missing_files else file_names,
                k=k
            )

            if missing_files:
                return self._format_response(
                    status="partial_missing",
                    retriever=retriever,
                    existing=existing_files,
                    missing=missing_files
                )

            return self._format_response(
                status="all_present",
                retriever=retriever,
                existing=existing_files
            )

        except Exception as e:
            print(f"[Pinecone] Error in multi retriever: {e}")
            return self._format_response(status="error")

class RetrieverService:

    def __init__(self, file_paths, user_id, config, api_key):
        self.file_paths = file_paths
        self.user_id = str(user_id)
        self.config = config
        self.api_key = api_key

        self.file_names = self._extract_file_names()

        self.pm = PineconeManager(
            index_name='rag-agent',
            config=config
        )

    # -----------------------------------
    # PRIVATE: Extract file names
    # -----------------------------------
    def _extract_file_names(self):
        import os
        return [os.path.basename(path) for path in self.file_paths]

    # -----------------------------------
    # PRIVATE: Get missing file paths
    # -----------------------------------
    def _get_missing_paths(self, missing_files):
        import os
        return [
            path for path in self.file_paths
            if os.path.basename(path) in missing_files
        ]

    # -----------------------------------
    # PRIVATE: Ingest missing files
    # -----------------------------------
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

    # -----------------------------------
    # PUBLIC: Get final retriever
    # -----------------------------------
    def get_retriever(self,chunk: int, overlap: int, sep: list, batch_size: int):
        try:
            result = self.pm.get_retriever_multi(
                user_id=self.user_id,
                file_names=self.file_names
            )

            # handle missing files
            if result["status"] in ["missing_all", "partial_missing"]:

                missing_files = result["missing_files"]
                missing_paths = self._get_missing_paths(missing_files)

                self._ingest_files(missing_paths=missing_paths,
                                   chunk=chunk, overlap=overlap, sep=sep, batch_size=batch_size)

                # re-fetch after ingestion
                result = self.pm.get_retriever_multi(
                    user_id=self.user_id,
                    file_names=self.file_names
                )

            return result.get("retriever", None)

        except Exception as e:
            print(f"[RetrieverService] Error: {e}")
            return None
# ========================== MAIN ============================

if __name__ == "__main__":

    from LLMUtils.LLMConfigs import GeminiConfig, api_key

    config = GeminiConfig(
        chat_model_name="gemini-3-flash-preview",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        temperature=0,
        top_p=0.8,
        top_k=32,
        max_output_tokens=3000,
        generation_max_tokens=8192,
        api_key=api_key
    )

    file_paths = [
    "E:/Tender Project/TATAAGM.pdf",
    "E:/Tender Project/MCA2.pdf",
    "E:/Tender Project/RILAGM.pdf",
    "E:/Tender Project/MCA.pdf",
    "E:/Tender Project/MCA1.pdf"
    
    ]

    service = RetrieverService(
        file_paths=file_paths,
        user_id=100,
        config=config,
        api_key=api_key
    )

    retriever = service.get_retriever(chunk=1500, overlap=250, sep=["\n\n", "\n", " ", ""], batch_size=10)

    print(retriever)

    