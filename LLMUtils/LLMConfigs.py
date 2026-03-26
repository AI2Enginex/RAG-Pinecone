
from typing import List, Optional
from typing import Annotated
import pandas as pd
from typing_extensions import TypedDict, Any
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI 
import os
# Loading the API key using lod_dotenv
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Loading the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Configuring Google Generative AI module with the provided API key
genai.configure(api_key=api_key)



# ========================== GEMINI CONFIG ============================

class GeminiConfig:
    """
    Stores all parameters for Gemini models (Chat & Embeddings).
    """
    def __init__(
        self,
        chat_model_name: str,
        embedding_model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_output_tokens: int,
        generation_max_tokens: int,
        api_key: str = None
    ):
        self.chat_model_name = chat_model_name
        self.embedding_model_name = embedding_model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.generation_max_tokens = generation_max_tokens
        self.api_key = api_key  # optional, for Gemini API calls


# ========================== QA STATE ============================
class QAState(TypedDict):
    question: str
    retrieved_chunks: List[str]
    answer: str
    prompt_type: Optional[str]
    next_action: Optional[str]
    verified: bool
    retriever: Any 

# ========================== GEMINI MODELS ============================

class GeminiModel:
    """Generic Gemini model wrapper using Google Generative AI SDK."""
    def __init__(self, config: GeminiConfig):
        try:
            genai.configure(api_key=config.api_key)
            self.config = config
            self.model = genai.GenerativeModel(self.config.chat_model_name)
            self.generation_config = genai.GenerationConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=self.config.generation_max_tokens,
            )
        except Exception as e:
            print(f"Error initializing GeminiModel: {e}")
            self.model = None
            self.generation_config = None

class GeminiChatModel(GeminiModel):
    """Wrapper to start a Gemini chat session."""
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        try:
            self.chat = self.model.start_chat() if self.model else None
        except Exception as e:
            print(f"Error initializing GeminiChatModel: {e}")
            self.chat = None

class ChatGoogleGENAI:
    """
    Wrapper for ChatGoogleGenerativeAI (LangChain integration for Gemini).
    """
    def __init__(self, config: GeminiConfig):
        try:
            self.config = config
            self.llm = ChatGoogleGenerativeAI(
                temperature=config.temperature,
                model=config.chat_model_name,
                google_api_key=config.api_key,
                top_p=config.top_p,
                top_k=config.top_k,
                max_output_tokens=config.max_output_tokens,
            )
            print("ChatGoogleGENAI initialized successfully.")
        except Exception as e:
            print(f"Error initializing ChatGoogleGENAI: {e}")
            self.llm = None

class EmbeddingModel:
    """Wrapper for HuggingFace embedding model."""
    def __init__(self, config: GeminiConfig):
        try:
            self.config = config
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model_name
            )
        except Exception as e:
            print(f"Error initializing EmbeddingModel: {e}")
            self.embeddings = None


if __name__ == "__main__":

    pass