from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from LLMUtils.LLMConfigs import EmbeddingModel

import os
from dotenv import load_dotenv

load_dotenv()
index_name = os.getenv("INDEX_NAME")
pinecone_api = os.getenv("PINECONE_API")

# ========================== VECTOR STORE ============================

class Vectors:
    """
    Handles generating vector embeddings and storing them in FAISS.
    """
    embeddings = None

    @classmethod
    def initialize(cls, config=None):
        try:
            cls.embeddings = EmbeddingModel(config=config).embeddings
            if cls.embeddings:
                print(f"Embedding model loaded: {config.embedding_model_name}")
            else:
                print("Embedding model failed to load.")
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API"))
            cls.index = pc.Index(index_name)

            return cls.embeddings
        except Exception as e:
            print(f"Failed to initialize embeddings: {e}")
            cls.embeddings = None
            cls.index = None

    @classmethod
    def generate_vectors_from_documents(cls, chunks: list, user_id: int, batch_size: int):

        vectors_to_upsert = list()
        try:
            if cls.embeddings is None:
                print("Embedding model not initialized.")
                return None
            
            if cls.index is None:
                print("Pinecone index not initialized.")
                return None
            
            if not chunks:
                print("No chunks provided for vector generation.")
                return None

            for doc in chunks:
                embedding = cls.embeddings.embed_query(doc.page_content)
            

                vectors_to_upsert.append((
                        f"{user_id}_{doc.metadata['chunk_id']}",   # unique ID
                        embedding,
                        {
                        **doc.metadata,
                        "user_id": str(user_id),        #add user_id
                        "text": doc.page_content        # store actual text
                    }
                    ))
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                cls.index.upsert(vectors=batch)

            print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone.")

            return cls.index
        except Exception as e:
            print(f"Error generating vectors: {e}")
            return None

if __name__ == "__main__":

    pass