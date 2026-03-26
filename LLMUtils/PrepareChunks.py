from langchain_text_splitters import RecursiveCharacterTextSplitter
# ========================== TEXT CHUNKING ============================

class TextChunks:
    """
    Handles splitting text into smaller chunks for LLM/embedding processing.
    """
    text_splitter = None

    @classmethod
    def initialize(cls, separator=None, chunksize=None, overlap=None):
        try:
            cls.text_splitter = RecursiveCharacterTextSplitter(
                separators=separator,
                chunk_size=chunksize,
                chunk_overlap=overlap
            )
            print(f"Text splitter initialized (chunk={chunksize}, overlap={overlap})")
            return cls.text_splitter
        except Exception as e:
            print(f"Failed to initialize text splitter: {e}")
            cls.text_splitter = None