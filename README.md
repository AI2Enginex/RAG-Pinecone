# RAG-QA

A small Retrieval-Augmented Generation (RAG) question-answering project built with PDF ingestion, Pinecone vector storage, and Google Gemini / LangChain integration.

## Project Overview

This repository is designed to:
- ingest PDF documents,
- preprocess and clean the text,
- split content into smaller chunks,
- generate embeddings,
- store vectors in Pinecone,
- retrieve relevant chunks for user queries,
- answer questions using a Gemini-based language model.

## Key Components

- `QA/qa_rag.py` - Main QA pipeline and entrypoint for running the system.
- `FileIngest/ReadData.py` - Reads PDF files and extracts text page by page.
- `LLMUtils/LLMConfigs.py` - Defines Gemini chat and embedding model wrappers and shared configuration.
- `LLMUtils/PrepareChunks.py` - Configures text splitters for chunking long documents.
- `LLMUtils/TextProcessing.py` - Handles preprocessing, section detection, chunk creation, and retrieval pipeline.
- `PromptClass/PromptClass.py` - Stores prompt templates and prompt manager logic.
- `Vectors/VectoreStore.py` - Handles embedding generation and Pinecone upsert operations.
- `pineconeclass.py` - Example Pinecone utility script for index operations.

## Folder Structure

- `FileIngest/` - PDF ingestion utilities.
- `LLMUtils/` - LLM and preprocessing utilities.
- `PromptClass/` - Prompt templates and prompt management.
- `QA/` - QA system implementation and graph-based execution.
- `Vectors/` - Vector store integration and embedding helpers.
- `Files/` - Placeholder for files and document data.

## Requirements

This project uses the following libraries (inferred from imports):
- `python-dotenv`
- `PyPDF2`
- `cleantext`
- `pandas`
- `pinecone-client`
- `langchain-core`
- `langchain-pinecone`
- `langchain-text-splitters`
- `langchain-huggingface`
- `langchain-google-genai`
- `google.generativeai`
- `langgraph`

## Environment Variables

Create a `.env` file in the project root with at least:

```env
PINECONE_API=<your_pinecone_api_key>
INDEX_NAME=<your_pinecone_index_name>
GOOGLE_API_KEY=<your_google_api_key>
```

## Usage

1. Install dependencies. Example using `pip`:

```bash
pip install python-dotenv PyPDF2 cleantext pandas pinecone-client langchain langchain-huggingface langchain-google-genai google-generative-ai langgraph
```

2. Configure `.env` with your Pinecone and Google API keys.

3. Update `QA/qa_rag.py` input file paths and user ID as needed.

4. Run the QA system:

```bash
python QA/qa_rag.py
```

5. Enter a question at the prompt and review the generated answer.

## How It Works

1. `QASystemGraphExecution` initializes the QA system and prepares PDF files.
2. `RetrieverService` checks Pinecone for existing embeddings for each file.
3. Missing files are ingested by `PrepareText`, which reads PDFs, cleans text, detects sections, and creates chunks.
4. `Vectors.generate_vectors_from_documents()` generates embeddings and upserts them into Pinecone.
5. User questions are filtered to relevant files, then a retriever searches the Pinecone index.
6. The selected Gemini prompt template is applied and the LLM generates an answer.
7. The system can optionally verify the answer against retrieved content.

## Notes

- The project assumes a Pinecone index is already created and available.
- File-level filtering is performed by matching file names against the user query.
- `QA/qa_rag.py` includes a simple graph-based workflow using `langgraph`.
- `pineconeclass.py` contains a sample Pinecone delete operation and is not required for normal QA usage.

