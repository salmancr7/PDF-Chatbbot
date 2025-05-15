import os
import uuid
import shutil
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, AIMessage
import psycopg2
from dotenv import load_dotenv
import re
import pandas as pd
import io
# Import for Word documents
import docx2txt

# Load environment variables
load_dotenv()

# Configuration - Using environment variables
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_docs")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_stores")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_FREE_MODEL = os.getenv("USE_FREE_MODEL", "true").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
ADVANCED_MODEL = os.getenv("ADVANCED_MODEL", "llama3")

# Chunk settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "work-demo")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "abcd123")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="API for processing documents (PDF, Word, Excel) and answering questions using AI",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request and response
class QueryRequest(BaseModel):
    session_id: str
    question: str
    use_free_model: Optional[bool] = None
    model_name: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[str]] = None
    model_used: str
    response_time: float

class DocumentUploadResponse(BaseModel):
    session_id: str
    message: str
    preview: str
    document_size: int
    document_type: str
    chunks_created: int
    model_used: str
    processing_time: float

# Session storage
active_sessions = {}

# Function to get database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

def clean_text(text):
    """Clean extracted text to improve quality"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text

def extract_text_from_document(doc_path):
    """Extract text from PDF, Word document, or Excel file"""
    file_ext = os.path.splitext(doc_path)[1].lower()
    text = ""
    total_pages = 0
    doc_type = "document"
    
    try:
        if file_ext == '.pdf':
            # Extract from PDF
            pdf_reader = PdfReader(doc_path)
            total_pages = len(pdf_reader.pages)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            doc_type = "pdf"
                    
        elif file_ext in ['.docx', '.doc']:
            # Extract from Word document
            text = docx2txt.process(doc_path)
            # Estimate pages (approx 3000 chars per page)
            total_pages = max(1, len(text) // 3000)
            doc_type = "word"
            
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            # Extract from Excel or CSV
            if file_ext == '.csv':
                df = pd.read_csv(doc_path)
            else:
                df = pd.read_excel(doc_path)
                
            # Convert DataFrame to string representation
            buffer = io.StringIO()
            df.to_csv(buffer)
            text = buffer.getvalue()
            # Count sheets as pages for Excel
            total_pages = 1
            if file_ext in ['.xlsx', '.xls']:
                excel_file = pd.ExcelFile(doc_path)
                total_pages = len(excel_file.sheet_names)
            doc_type = "excel"
            
        else:
            return None, 0, "unsupported"
            
        # Clean the extracted text
        text = clean_text(text)
            
        if not text.strip():
            print(f"Warning: No text extracted from {file_ext} file.")
            
        return text, total_pages, doc_type
        
    except Exception as e:
        print(f"Error reading document: {str(e)}")
        return None, 0, None

def split_text_into_chunks(text):
    """Split text into chunks with improved strategy"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Create vector store with text embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def create_conversation_chain(vector_store, use_free_model=USE_FREE_MODEL, model_name=None):
    """Create an enhanced conversation chain for comprehensive Q&A"""
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )
    
    if use_free_model:
        try:
            model_to_use = model_name if model_name else ADVANCED_MODEL if ADVANCED_MODEL else OLLAMA_MODEL
            
            llm = OllamaLLM(
                model=model_to_use,
                base_url=OLLAMA_HOST,
                temperature=0.2
            )
            model_used = f"Ollama - {model_to_use}"
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            print("Falling back to Google Gemini")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
            model_used = "Google Gemini"
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
        model_used = "Google Gemini"
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    custom_template = """
    You are an expert AI assistant answering questions about this specific document.
    
    When answering questions:
    1. Base your answer DIRECTLY on information in the document
    2. If the exact answer isn't in the document context, clearly state this
    3. Use DIRECT QUOTES from the document whenever possible
    4. Organize information in a clear, structured way
    5. Maintain the terminology used in the original document
    6. Be precise with numbers, dates, and technical details from the document
    
    Chat History:
    {chat_history}
    
    Context from the document:
    {context}
    
    Question: {question}
    
    Provide a thorough answer that directly addresses the question.
    If creating tables, use proper HTML table tags.
    Always verify your answer against the document context before responding.
    """
    
    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=custom_template,
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=True,
    )
    
    return conversation_chain, model_used

# Function to get a session
def get_session(session_id: str, use_free_model=None, model_name=None):
    """Get conversation chain for the given session ID"""
    if session_id not in active_sessions:
        # Try to load from disk if exists
        vector_store_path = os.path.join(VECTOR_STORE_DIR, session_id)
        if os.path.exists(vector_store_path):
            try:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.load_local(vector_store_path, embeddings)
                
                use_model = USE_FREE_MODEL if use_free_model is None else use_free_model
                model_to_use = model_name if model_name else None
                
                conversation_chain, model_used = create_conversation_chain(vector_store, use_model, model_to_use)
                
                active_sessions[session_id] = {
                    "conversation": conversation_chain,
                    "use_free_model": use_model,
                    "model_name": model_to_use,
                    "model_used": model_used
                }
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Session not found or couldn't be loaded: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    elif (use_free_model is not None and use_free_model != active_sessions[session_id].get("use_free_model", USE_FREE_MODEL)) or \
         (model_name is not None and model_name != active_sessions[session_id].get("model_name")):
        # Model preference changed, recreate conversation chain
        vector_store_path = os.path.join(VECTOR_STORE_DIR, session_id)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(vector_store_path, embeddings)
        
        conversation_chain, model_used = create_conversation_chain(vector_store, use_free_model, model_name)
        
        active_sessions[session_id]["conversation"] = conversation_chain
        active_sessions[session_id]["use_free_model"] = use_free_model
        active_sessions[session_id]["model_name"] = model_name
        active_sessions[session_id]["model_used"] = model_used
    
    return active_sessions[session_id]

# API Endpoints
@app.post("/upload-document/", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    use_free_model: bool = Form(USE_FREE_MODEL),
    model_name: Optional[str] = Form(None)
):
    """
    Upload and process a document (PDF, Word, Excel)
    """
    start_time = time.time()
    
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    supported_formats = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.csv']
    
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Please upload a PDF, Word, or Excel file. Supported formats: {', '.join(supported_formats)}"
        )
    
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Save the uploaded document
        doc_path = os.path.join(UPLOAD_DIR, f"{session_id}{file_ext}")
        with open(doc_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        raw_text, total_pages, doc_type = extract_text_from_document(doc_path)
        if not raw_text:
            raise HTTPException(status_code=500, detail=f"Failed to extract text from {doc_type.upper()} file")
            
        text_chunks = split_text_into_chunks(raw_text)
        
        # Create and save vector store
        vector_store = create_vector_store(text_chunks)
        vector_store_path = os.path.join(VECTOR_STORE_DIR, session_id)
        vector_store.save_local(vector_store_path)
        
        # Create conversation chain with model preference
        conversation_chain, model_used = create_conversation_chain(vector_store, use_free_model, model_name)
        
        # Store the conversation chain in memory
        active_sessions[session_id] = {
            "conversation": conversation_chain,
            "doc_path": doc_path,
            "doc_type": doc_type,
            "use_free_model": use_free_model,
            "model_name": model_name,
            "model_used": model_used
        }
        
        # Store session in database
        conn = get_db_connection()
        try:
            conn.autocommit = False
            cursor = conn.cursor()
            
            # Check if table structure supports additional columns
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'sessions' AND column_name = 'total_pages'
                )
            """)
            column_exists = cursor.fetchone()[0]

            if column_exists:
                cursor.execute(
                    """INSERT INTO sessions 
                    (session_id, pdf_path, total_pages, total_chunks, llm_model) 
                    VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, doc_path, total_pages, len(text_chunks), model_used)
                )
            else:
                cursor.execute(
                    """INSERT INTO sessions 
                    (session_id, pdf_path, llm_model) 
                    VALUES (%s, %s, %s)""",
                    (session_id, doc_path, model_used)
                )
                
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        # Create a preview of the document content
        preview = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            session_id=session_id,
            message=f"{doc_type.upper()} processed successfully with {len(text_chunks)} chunks",
            preview=preview,
            document_size=len(raw_text),
            document_type=doc_type.upper(),
            chunks_created=len(text_chunks),
            model_used=model_used,
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        file.file.close()

@app.post("/query/", response_model=QueryResponse)
async def query_document(request: QueryRequest = Body(...)):
    """
    Ask a question about a processed document
    """
    start_time = time.time()
    
    try:
        # Get session with optional model preference change
        session = get_session(request.session_id, request.use_free_model, request.model_name)
        conversation = session["conversation"]
        model_used = session["model_used"]
        
        # Process the question
        response = conversation({"question": request.question})
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Store the conversation in the database
        conn = get_db_connection()
        try:
            conn.autocommit = False
            cursor = conn.cursor()
            
            # Store user question
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (request.session_id, "human", request.question)
            )
            
            # Get the full AI response text
            answer_text = response["answer"]
            
            # Store AI response with response time
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content, query_time) VALUES (%s, %s, %s, %s)",
                (request.session_id, "ai", answer_text, response_time)
            )
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        # Extract source document content for response
        source_docs = []
        if "source_documents" in response:
            for doc in response["source_documents"]:
                source_docs.append(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        
        return QueryResponse(
            answer=response["answer"],
            source_documents=source_docs,
            model_used=model_used,
            response_time=response_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """
    Get the conversation history for a session
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute("SELECT 1 FROM sessions WHERE session_id = %s", (session_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get conversation history with explicit ordering
        cursor.execute(
            """SELECT id, role, content, query_time, timestamp 
               FROM conversation_history 
               WHERE session_id = %s 
               ORDER BY id, timestamp""",
            (session_id,)
        )
        
        history = []
        for id, role, content, query_time, timestamp in cursor.fetchall():
            history.append({
                "id": id,
                "role": role,
                "content": content,
                "query_time": query_time,
                "timestamp": timestamp.isoformat() if timestamp else None
            })
        
        # Include model and document info if available
        model_used = "Unknown"
        doc_type = "Unknown"
        if session_id in active_sessions:
            if "model_used" in active_sessions[session_id]:
                model_used = active_sessions[session_id]["model_used"]
            if "doc_type" in active_sessions[session_id]:
                doc_type = active_sessions[session_id]["doc_type"].upper()
        
        return {
            "session_id": session_id, 
            "history": history,
            "model_used": model_used,
            "document_type": doc_type
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.get("/available-models")
async def get_available_models():
    """
    Get list of available models
    """
    try:
        # Basic free models always available
        models = [
            {"name": "mistral", "type": "free", "description": "Fast, good for general purpose use"},
            {"name": "llama3", "type": "free", "description": "Powerful general-purpose model"},
        ]
        
        # Try to get models from Ollama
        try:
            import requests
            response = requests.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                ollama_models = response.json().get("models", [])
                # Add any additional models found
                for model in ollama_models:
                    model_name = model.get("name")
                    if model_name and model_name not in [m["name"] for m in models]:
                        models.append({
                            "name": model_name,
                            "type": "free",
                            "description": f"Available on your Ollama instance"
                        })
        except Exception as e:
            print(f"Could not fetch Ollama models: {e}")
        
        # Add Google model
        models.append({
            "name": "gemini-1.5-pro",
            "type": "api",
            "description": "Google's most powerful model (requires API key)"
        })
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching available models: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported document formats
    """
    return {
        "formats": [
            {"extension": ".pdf", "type": "PDF", "description": "Portable Document Format"},
            {"extension": ".docx", "type": "Word", "description": "Microsoft Word Document"},
            {"extension": ".doc", "type": "Word", "description": "Microsoft Word Document (Legacy)"},
            {"extension": ".xlsx", "type": "Excel", "description": "Microsoft Excel Spreadsheet"},
            {"extension": ".xls", "type": "Excel", "description": "Microsoft Excel Spreadsheet (Legacy)"},
            {"extension": ".csv", "type": "CSV", "description": "Comma-Separated Values"}
        ]
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "ok", 
        "default_model": f"Ollama - {OLLAMA_MODEL}" if USE_FREE_MODEL else "Google Gemini",
        "supported_formats": ["PDF", "Word", "Excel", "CSV"],
        "chunk_settings": {
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "retrieval_k": RETRIEVAL_K
        }
    }