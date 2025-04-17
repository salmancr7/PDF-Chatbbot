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
# Updated imports for LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# Updated import for Ollama
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, AIMessage
import psycopg2
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configuration - Using environment variables
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_pdfs")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_stores")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_FREE_MODEL = os.getenv("USE_FREE_MODEL", "true").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
ADVANCED_MODEL = os.getenv("ADVANCED_MODEL", "llama3")  # More powerful option

# Chunk settings - critical for performance
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))  # Number of chunks to retrieve

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
    title="Enhanced PDF Q&A API",
    description="API for processing PDFs and answering questions using advanced AI models",
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

class PDFUploadResponse(BaseModel):
    session_id: str
    message: str
    preview: str
    document_size: int
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
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix broken words (some PDFs break words with hyphen at line end)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extract complete text from PDF file with improved robustness"""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"  # Add page breaks for better segmentation
        
        # Clean the extracted text
        text = clean_text(text)
        
        if not text.strip():
            print("Warning: No text was extracted from the PDF. The file might be scanned images or protected.")
            
        return text, total_pages
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return None, 0

def split_text_into_chunks(text):
    """Split text into chunks with improved strategy"""
    # Using optimized chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Better natural boundaries
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Create vector store with text embeddings"""
    # Use a robust embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def create_conversation_chain(vector_store, use_free_model=USE_FREE_MODEL, model_name=None):
    """Create an enhanced conversation chain for comprehensive Q&A"""
    
    # Set up retriever with optimized k value
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Use similarity search
        search_kwargs={"k": RETRIEVAL_K}  # Retrieve more chunks for better context
    )
    
    # Choose model based on parameters
    if use_free_model:
        try:
            # Select model to use - prefer specified model, then advanced, then default
            model_to_use = model_name if model_name else ADVANCED_MODEL if ADVANCED_MODEL else OLLAMA_MODEL
            
            # Connect to Ollama
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
        # Use Google's Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
        model_used = "Google Gemini"
    
    # Configure memory with explicit output key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Configure improved prompt template
    custom_template = """
    You are an expert AI assistant answering questions about this specific PDF document.
    
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
    
    # Create the conversation chain with the custom prompt
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
                
                # Use provided model preference or default
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
        
        # Update session with new model
        active_sessions[session_id]["conversation"] = conversation_chain
        active_sessions[session_id]["use_free_model"] = use_free_model
        active_sessions[session_id]["model_name"] = model_name
        active_sessions[session_id]["model_used"] = model_used
    
    return active_sessions[session_id]

# API Endpoints
@app.post("/upload-pdf/", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    use_free_model: bool = Form(USE_FREE_MODEL),
    model_name: Optional[str] = Form(None)
):
    """
    Upload and process a PDF file
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Save the uploaded PDF
        pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        raw_text, total_pages = extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF")
            
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
            "pdf_path": pdf_path,
            "use_free_model": use_free_model,
            "model_name": model_name,
            "model_used": model_used
        }
        
        # Store session in database
        conn = get_db_connection()
        try:
            conn.autocommit = False  # Use transaction for reliability
            cursor = conn.cursor()
            
            # Check if the table exists and if total_pages column exists
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
                    (session_id, pdf_path, total_pages, len(text_chunks), model_used)
                )
            else:
                # Fallback to a simpler query if the column doesn't exist
                cursor.execute(
                    """INSERT INTO sessions 
                    (session_id, pdf_path, llm_model) 
                    VALUES (%s, %s, %s)""",
                    (session_id, pdf_path, model_used)
                )
                
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        # Create a preview of the PDF content
        preview = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        
        processing_time = time.time() - start_time
        
        return PDFUploadResponse(
            session_id=session_id,
            message=f"PDF processed successfully with {len(text_chunks)} chunks",
            preview=preview,
            document_size=len(raw_text),
            chunks_created=len(text_chunks),
            model_used=model_used,
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        file.file.close()

@app.post("/query/", response_model=QueryResponse)
async def query_pdf(request: QueryRequest = Body(...)):
    """
    Ask a question about a processed PDF
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
            # Use transaction for reliable storage
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
            
            # Explicitly commit the transaction
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
        
        # Include model info if available
        model_used = "Unknown"
        if session_id in active_sessions and "model_used" in active_sessions[session_id]:
            model_used = active_sessions[session_id]["model_used"]
        
        return {
            "session_id": session_id, 
            "history": history,
            "model_used": model_used
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

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "ok", 
        "default_model": f"Ollama - {OLLAMA_MODEL}" if USE_FREE_MODEL else "Google Gemini",
        "chunk_settings": {
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "retrieval_k": RETRIEVAL_K
        }
    }