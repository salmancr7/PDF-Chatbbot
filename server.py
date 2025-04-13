import os
import uuid
import shutil
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama  # Added for free model support
from langchain.schema import HumanMessage, AIMessage
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration - Using environment variables
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_pdfs")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_stores")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_FREE_MODEL = os.getenv("USE_FREE_MODEL", "false").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "pdf_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Q&A API",
    description="API for processing PDFs and answering questions using AI models",
    version="1.0.0"
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
    use_free_model: Optional[bool] = None  # Make it optional to override default

class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[str]] = None
    model_used: str

class PDFUploadResponse(BaseModel):
    session_id: str
    message: str
    preview: str
    document_size: int
    chunks_created: int
    model_used: str

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

def extract_text_from_pdf(pdf_path):
    """Extract complete text from PDF file at the given path"""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:  # Only add if text was successfully extracted
                text += page_text + "\n\n"  # Add page breaks for better segmentation
        
        if not text.strip():
            print("Warning: No text was extracted from the PDF. The file might be scanned images or protected.")
            
        return text
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return None

def split_text_into_chunks(text):
    """Split text into manageable chunks with appropriate overlap for context preservation"""
    # Use a smaller chunk size with more overlap to ensure no content is missed
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Smaller chunks ensure more granular retrieval
        chunk_overlap=200,     # Good overlap preserves context between chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Try to split at natural boundaries first
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Create vector store with text embeddings"""
    # Use a free local embedding model instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def create_conversation_chain(vector_store, use_free_model=USE_FREE_MODEL):
    """Create an enhanced conversation chain for comprehensive Q&A across the entire document"""
    
    # Set up a more effective retriever with higher top_k to get more context
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Use similarity search
        search_kwargs={
            "k": 5  # Retrieve more chunks for better context
        }
    )
    
    # Choose between Google Gemini or Ollama (free, local model)
    if use_free_model:
        try:
            # Try to use local Ollama with Mistral model
            llm = Ollama(
                model="mistral",  # A powerful open-source model
                base_url=OLLAMA_HOST,
                temperature=0.2
            )
            print("Using FREE Mistral model via Ollama")
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            print("Falling back to Google Gemini")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
    else:
        # Use Google's Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
    
    # Configure memory with explicit output key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Configure custom prompt to encourage comprehensive answers with supplementary info
    custom_template = """
    You are an assistant that answers questions based on documents.
    
    Chat History:
    {chat_history}
    
    Context from the document:
    {context}
    
    Question: {question}
    
    Answer the question as completely as possible using the provided context.
    If the answer is not fully covered in the context, you can add supplementary information to
    make your answer more helpful. In these cases, clearly indicate what information comes
    from outside the document by prefacing with "Additionally:" or similar phrasing.
    
    Provide a complete, informative answer that is useful to the user.
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
        return_source_documents=True,  # Returns source chunks for verification
    )
    
    return conversation_chain, "Mistral (Free)" if use_free_model else "Google Gemini"

# Function to get a session
def get_session(session_id: str, use_free_model=None):
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
                conversation_chain, model_used = create_conversation_chain(vector_store, use_model)
                
                active_sessions[session_id] = {
                    "conversation": conversation_chain,
                    "use_free_model": use_model,
                    "model_used": model_used
                }
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Session not found or couldn't be loaded: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    elif use_free_model is not None and use_free_model != active_sessions[session_id].get("use_free_model", USE_FREE_MODEL):
        # Model preference changed, recreate conversation chain
        vector_store_path = os.path.join(VECTOR_STORE_DIR, session_id)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(vector_store_path, embeddings)
        
        conversation_chain, model_used = create_conversation_chain(vector_store, use_free_model)
        
        # Update session with new model
        active_sessions[session_id]["conversation"] = conversation_chain
        active_sessions[session_id]["use_free_model"] = use_free_model
        active_sessions[session_id]["model_used"] = model_used
    
    return active_sessions[session_id]

# API Endpoints
@app.post("/upload-pdf/", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    use_free_model: bool = Form(USE_FREE_MODEL)  # Default from environment
):
    """
    Upload and process a PDF file
    """
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
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF")
            
        text_chunks = split_text_into_chunks(raw_text)
        
        # Create and save vector store
        vector_store = create_vector_store(text_chunks)
        vector_store_path = os.path.join(VECTOR_STORE_DIR, session_id)
        vector_store.save_local(vector_store_path)
        
        # Create conversation chain with model preference
        conversation_chain, model_used = create_conversation_chain(vector_store, use_free_model)
        
        # Store the conversation chain in memory
        active_sessions[session_id] = {
            "conversation": conversation_chain,
            "pdf_path": pdf_path,
            "use_free_model": use_free_model,
            "model_used": model_used
        }
        
        # Store session in database
        conn = get_db_connection()
        try:
            conn.autocommit = False  # Use transaction for reliability
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, pdf_path) VALUES (%s, %s)",
                (session_id, pdf_path)
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
        
        return PDFUploadResponse(
            session_id=session_id,
            message=f"PDF processed successfully with {len(text_chunks)} chunks",
            preview=preview,
            document_size=len(raw_text),
            chunks_created=len(text_chunks),
            model_used=model_used
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
    try:
        # Get session with optional model preference change
        session = get_session(request.session_id, request.use_free_model)
        conversation = session["conversation"]
        model_used = session["model_used"]
        
        # Process the question
        response = conversation({"question": request.question})
        
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
            print(f"Storing AI response of length: {len(answer_text)} characters")
            
            # Store AI response - ensure we're storing the complete answer
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (request.session_id, "ai", answer_text)
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
            model_used=model_used
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
            "SELECT id, role, content, timestamp FROM conversation_history WHERE session_id = %s ORDER BY id, timestamp",
            (session_id,)
        )
        
        history = []
        for id, role, content, timestamp in cursor.fetchall():
            history.append({
                "id": id,
                "role": role,
                "content": content,  # The full content will be retrieved properly
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

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok", "default_model": "Mistral (Free)" if USE_FREE_MODEL else "Google Gemini"}