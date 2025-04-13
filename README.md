PDF AI Assistant
A powerful AI-powered application for interrogating PDF documents, built with LangChain, Streamlit, FastAPI, and Google's Gemini or Mistral AI models.
Features

Dual-Interface: Available as both a Streamlit web application and a FastAPI backend
Model Flexibility: Use Google's Gemini or the free local Mistral model via Ollama
Comprehensive PDF Processing: Extracts text from PDFs and creates semantic embeddings
Intelligent Q&A: Ask questions about PDF content and get AI-powered answers with source references
Conversational Memory: Remembers previous questions for context-aware responses
Database Integration: Stores conversations and sessions in PostgreSQL
Advanced UI: Dark-themed UI with custom styling (Streamlit version)
REST API: Complete API for PDF processing and querying (FastAPI version)

Project Structure

pdf-ai-assistant/
├── main.py                # Streamlit web application
├── server.py              # FastAPI backend server
├── db_init.py             # Database initialization script
├── .env                   # Environment variables configuration
├── requirements.txt       # Python dependencies
├── uploaded_pdfs/         # Storage for uploaded PDFs
└── vector_stores/         # FAISS vector database storage

Setup Instructions
Prerequisites

Python 3.9+ installed
PostgreSQL database server
(Optional) Ollama for running Mistral locally

Install required packages:
install -r requirements.txt

Set up your PostgreSQL database
 Create a database named 'work-demo' (or whatever you set in .env)
createdb work-demo

# Initialize the database tables
python db_init.py

Configure environment variables:

Copy or rename the .env.example file to .env
Edit the .env file with your API keys and configuration


(Optional) Install and run Ollama for local model support
# Install Ollama (refer to https://ollama.ai for platform-specific instructions)

# Run the Mistral model
ollama pull mistral
ollama run mistral

Running the Application
Streamlit Web Application:
streamlit run main.py

FastAPI Backend Server:
uvicorn server:app --reload


Configuration Options (.env file)

# API Configuration
GOOGLE_API_KEY=your_api_key_here  # Required for Google Gemini model

# Use free model option (true/false)
USE_FREE_MODEL=true
OLLAMA_HOST=http://localhost:11434

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=work-demo
DB_USER=postgres
DB_PASSWORD=your_password

# Storage Paths
UPLOAD_DIR=uploaded_pdfs
VECTOR_STORE_DIR=vector_stores

# PDF Path (for Streamlit app)
PDF_PATH=path/to/your/document.pdf

Detailed Component Documentation
1. Streamlit Web Application (main.py)
The Streamlit application provides a user-friendly web interface for interacting with PDFs through AI.
Key functionality:

PDF document loading and processing
Text extraction and chunking
Vector database creation with FAISS
Chat interface for asking questions about the document
Displays answer sources and references
Option to switch between Gemini and Mistral models
Dark mode UI with custom styling
Database integration for conversation persistence

Usage:

Click "Process PDF Document" to analyze the PDF
Enter questions in the text input area
View AI responses and source references
(Optional) Toggle between free and paid AI models

FastAPI Backend (server.py)
The FastAPI backend provides a REST API for programmatic access to the PDF AI capabilities.
Key endpoints:

POST /upload-pdf/: Upload and process a PDF file
POST /query/: Ask a question about a processed PDF
GET /sessions/{session_id}/history: Retrieve conversation history
GET /health: Health check endpoint

API Features:

Stateless API design with session management
Model switching capability
PostgreSQL integration for persistent storage
Comprehensive error handling
CORS support for cross-origin requests

3. Database Initialization (db_init.py)
Sets up the required PostgreSQL tables for the application:

sessions: Stores information about uploaded PDFs
conversation_history: Stores the conversation exchanges

4. Common Components
Both implementations share key processing components:
PDF Processing Pipeline:

Text Extraction: Uses PyPDF2 to extract text from PDF documents
Text Chunking: Splits large texts into manageable chunks using RecursiveCharacterTextSplitter
Vector Embedding: Creates embeddings using HuggingFace's all-MiniLM-L6-v2 model
Vector Storage: Stores vectors in a FAISS database for efficient similarity search


AI Model Support:

Google Gemini: High-quality commercial model via API
Mistral: Free, locally-hosted alternative via Ollama

Question Answering System:

Uses LangChain's ConversationalRetrievalChain
Custom prompts for comprehensive answers
Source document retrieval and citation
Conversation memory for contextual follow-ups

Usage Examples
Streamlit Interface
Ask factual questions about your document:

Troubleshooting
Common Issues

PDF Processing Fails:

Ensure the PDF is not password-protected
Check if the PDF contains actual text (not just scanned images)


Database Connection Issues:

Verify PostgreSQL is running
Check credentials in .env file
Ensure the database exists


Model Errors:

For Google Gemini: Verify your API key
For Mistral: Ensure Ollama is running and the model is installed


Out of Memory Errors:

Reduce chunk size in the configuration
Process smaller PDFS
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


Acknowledgements

LangChain for the excellent framework
Google for the Gemini AI model
Mistral AI for the open-source model
Ollama for local model hosting
Streamlit and FastAPI for application frameworks