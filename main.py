import os
import uuid
import argparse
import streamlit as st
import html
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated imports for LangChain components

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
# Updated import for Ollama
from langchain_ollama import OllamaLLM
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import re
import time

# Load environment variables
load_dotenv()

# Configuration variables
PDF_PATH = os.getenv("PDF_PATH", r"C:\Users\PRECISION 7770\Downloads\RFP for Integrated Labour Management System - ver 21.2.pdf")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_FREE_MODEL = os.getenv("USE_FREE_MODEL", "true").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # Default model

# Optional advanced models if available
ADVANCED_MODEL = os.getenv("ADVANCED_MODEL", "llama3")  # llama3, mixtral, or other advanced models

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

# Set page configuration
st.set_page_config(
    page_title="Enhanced PDF AI Assistant", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Enhanced PDF AI Assistant\nA powerful tool to interact with PDF documents using AI."
    }
)

# Custom CSS (keeping your existing style)
st.markdown("""
<style>
    .main {
        background-color: #1e1e2e;
        color: white;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .user-message {
        background-color: #2C3E50;
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .ai-message {
        background-color: #3498DB;
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metrics-card {
        background-color: #2C3E50;
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .css-1v3fvcr {
        background-color: #1e1e2e;
    }
    .stTextInput input {
        background-color: #2C3E50;
        color: white;
        border: 1px solid #3498DB;
    }
    .streamlit-expanderHeader {
        background-color: #2C3E50;
        color: white;
    }
    .stMarkdown, h1, h2, h3, p, span {
        color: white;
    }
    .ai-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    .ai-table th, .ai-table td {
        border: 1px solid #4a6572;
        padding: 8px 12px;
        text-align: left;
    }
    .ai-table th {
        background-color: #2C3E50;
        color: white;
    }
    .ai-table tr {
        background-color: #34495e;
        color: white;
    }
    .ai-table tr:nth-child(even) {
        background-color: #3d566e;
    }
    .feedback-buttons {
        display: flex;
        justify-content: flex-end;
        margin-top: 5px;
    }
    .feedback-button {
        background-color: transparent;
        color: white;
        border: 1px solid #3498DB;
        padding: 5px 10px;
        margin-left: 5px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = {"chars": 0, "chunks": 0, "pages": 0}
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "use_free_model" not in st.session_state:
    st.session_state.use_free_model = USE_FREE_MODEL
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = OLLAMA_MODEL
if "response_times" not in st.session_state:
    st.session_state.response_times = []
# Add these variables to track settings changes
if "current_chunk_size" not in st.session_state:
    st.session_state.current_chunk_size = CHUNK_SIZE
if "current_chunk_overlap" not in st.session_state:
    st.session_state.current_chunk_overlap = CHUNK_OVERLAP
if "current_retrieval_k" not in st.session_state:
    st.session_state.current_retrieval_k = RETRIEVAL_K

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
        
        if not text.strip():
            print("Warning: No text was extracted from the PDF. The file might be scanned images or protected.")
            
        return text, total_pages
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return None, 0

def clean_text(text):
    """Clean extracted text to improve quality"""
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix broken words (some PDFs break words with hyphen at line end)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    return text

def split_text_into_chunks(text):
    """Split text into manageable chunks with improved strategy"""
    # Using improved chunk size and overlap for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.current_chunk_size,
        chunk_overlap=st.session_state.current_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Better natural boundaries
    )
    
    chunks = text_splitter.split_text(text)
    
    # Verify we're getting reasonable chunks
    if chunks:
        avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
        print(f"Created {len(chunks)} chunks with average size of {avg_chunk_size:.1f} characters")
    else:
        print("Warning: No chunks were created. Input text might be empty.")
    
    return chunks

def create_vector_store(text_chunks):
    """Create vector store with text embeddings"""
    try:
        # Use a robust embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create the vector store
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def create_conversation_chain(vector_store):
    """Create an enhanced conversation chain for comprehensive Q&A"""
    
    # Set up a more effective retriever with optimized k value
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": st.session_state.current_retrieval_k  # Retrieve more chunks for better context
        }
    )
    
    # Choose between Google Gemini or Ollama (free models)
    if st.session_state.use_free_model:
        try:
            # Try to use advanced model if available, otherwise fallback to mistral
            model_to_use = ADVANCED_MODEL if ADVANCED_MODEL and ADVANCED_MODEL != "mistral" else st.session_state.ollama_model
            
            # Connect to Ollama
            llm = OllamaLLM(
                model=model_to_use,
                base_url=OLLAMA_HOST,
                temperature=0.2
            )
            print(f"Using FREE model: {model_to_use} via Ollama")
            st.session_state.model_used = f"Ollama - {model_to_use}"
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            st.warning("Failed to connect to Ollama. Falling back to Google Gemini.")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
            st.session_state.model_used = "Google Gemini"
    else:
        # Use Google's Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
        st.session_state.model_used = "Google Gemini"
    
    # Configure memory with the explicit output key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Configure improved prompt template for better answers
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
    If creating tables, use proper HTML table tags with the "ai-table" class.
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
        verbose=True
    )
    
    return conversation_chain

def sanitize_ai_response(response_text):
    """Clean AI response to prevent HTML rendering issues"""
    # Remove any stray closing div tags
    response_text = re.sub(r'</div>', '', response_text)
    
    # Ensure proper table formatting if tables are present
    if "<table" in response_text.lower():
        # Add table styling class
        response_text = re.sub(r'<table', '<table class="ai-table"', response_text, flags=re.IGNORECASE)
        
        # Make sure all table tags are properly closed
        if response_text.count("<table") > response_text.count("</table"):
            response_text += "</table>"
    
    # Check for other unclosed HTML tags
    common_tags = ["div", "p", "span", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
    for tag in common_tags:
        opening_count = len(re.findall(f"<{tag}[^>]*>", response_text, re.IGNORECASE))
        closing_count = len(re.findall(f"</{tag}>", response_text, re.IGNORECASE))
        
        # Add missing closing tags if needed
        if opening_count > closing_count:
            for _ in range(opening_count - closing_count):
                response_text += f"</{tag}>"
    
    return response_text

def process_user_question(user_question):
    """Process user question with improved accuracy and response tracking"""
    if st.session_state.conversation is None:
        st.error("Please process the PDF document first.")
        return
    
    # Start timing the response
    start_time = time.time()
    
    # Load conversation history from database
    if st.session_state.session_id:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM conversation_history WHERE session_id = %s ORDER BY timestamp",
                (st.session_state.session_id,)
            )
            
            # Convert to LangChain message format
            messages = []
            for role, content in cursor.fetchall():
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            
            # Replace conversation memory with database history
            if messages and hasattr(st.session_state.conversation, 'memory'):
                st.session_state.conversation.memory.chat_memory.messages = messages
        except Exception as e:
            st.error(f"Error loading conversation history: {str(e)}")
        finally:
            conn.close()
    
    # Process the question
    response = st.session_state.conversation({"question": user_question})
    
    # Calculate response time
    response_time = time.time() - start_time
    st.session_state.response_times.append(response_time)
    
    # Get and sanitize the AI response
    answer_text = sanitize_ai_response(response["answer"])
    
    # Store the new messages in the database
    if st.session_state.session_id:
        conn = get_db_connection()
        try:
            # Use transaction for reliable storage
            conn.autocommit = False
            cursor = conn.cursor()
            
            # Store user question
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (st.session_state.session_id, "human", user_question)
            )
            
            # Store AI response with response time
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content, query_time) VALUES (%s, %s, %s, %s)",
                (st.session_state.session_id, "ai", answer_text, response_time)
            )
            
            # Explicitly commit the transaction
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            st.error(f"Error storing conversation: {str(e)}")
        finally:
            cursor.close()
            conn.close()
    
    # Update the chat history with the sanitized response
    history = response["chat_history"]
    if len(history) > 0 and isinstance(history[-1], AIMessage):
        history[-1] = AIMessage(content=answer_text)
    
    st.session_state.chat_history = history
    
    return answer_text, response.get("source_documents", []), response_time

def process_pdf():
    """Process the PDF file with improved text extraction and chunking"""
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found at: {PDF_PATH}")
        return False
    
    with st.spinner("Processing PDF - this may take a while for large documents..."):
        # Extract text from PDF
        progress_text = st.empty()
        progress_text.text("Extracting text from PDF...")
        
        raw_text, total_pages = extract_text_from_pdf(PDF_PATH)
        
        if not raw_text:
            st.error("Failed to extract text from the PDF. The file might be protected or contain only images.")
            return False
        
        # Add a progress indicator
        progress_bar = st.progress(0)
        
        # Clean the extracted text
        progress_text.text("Cleaning and preparing text...")
        progress_bar.progress(20)
        cleaned_text = clean_text(raw_text)
        
        # Show PDF content length information
        progress_text.text(f"Extracted {len(cleaned_text)} characters of text from {total_pages} pages")
        
        # Process the extracted text
        progress_text.text("Splitting text into optimal chunks...")
        progress_bar.progress(40)
        text_chunks = split_text_into_chunks(cleaned_text)
        
        # Create vector store
        progress_text.text("Creating vector database with embeddings...")
        progress_bar.progress(60)
        st.session_state.vector_store = create_vector_store(text_chunks)
        
        # Create conversation chain
        progress_text.text("Setting up Q&A system with enhanced prompts...")
        progress_bar.progress(80)
        st.session_state.conversation = create_conversation_chain(st.session_state.vector_store)
        
        # Create session in database
        session_id = str(uuid.uuid4())
        conn = get_db_connection()
        try:
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
                    (session_id, PDF_PATH, total_pages, len(text_chunks), st.session_state.model_used)
                )
            else:
                # Fallback to a simpler query if the column doesn't exist
                cursor.execute(
                    """INSERT INTO sessions 
                       (session_id, pdf_path, llm_model) 
                       VALUES (%s, %s, %s)""",
                    (session_id, PDF_PATH, st.session_state.model_used)
                )
            conn.commit()
            st.session_state.session_id = session_id
        except Exception as e:
            st.error(f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        progress_bar.progress(100)
        progress_text.text("PDF processing complete!")
        st.session_state.pdf_processed = True
        
        # Save stats
        st.session_state.pdf_stats = {
            "chars": len(cleaned_text),
            "chunks": len(text_chunks),
            "pages": total_pages,
            "chunk_size": st.session_state.current_chunk_size,
            "chunk_overlap": st.session_state.current_chunk_overlap
        }
        
        return True
    
    return False

def main():
    # Main layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
    
    with col2:
        st.title("Enhanced PDF AI Assistant")
        st.markdown("Ask detailed questions about your PDF document and get accurate AI-powered answers.")
    
    # Sidebar for configuration and controls
    st.sidebar.header("📋 Document Control")
    
    # Model selection options
    st.session_state.use_free_model = st.sidebar.checkbox(
        "Use free local model",
        value=st.session_state.use_free_model,
        help="Use locally hosted models via Ollama instead of Google Gemini"
    )
    
    if st.session_state.use_free_model:
        # Ollama model selection
        available_models = ["mistral", "llama3", "mixtral", "phi3", "gemma"]
        selected_model = st.sidebar.selectbox(
            "Select Ollama Model",
            available_models,
            index=available_models.index(st.session_state.ollama_model) if st.session_state.ollama_model in available_models else 0,
            help="Choose which model to use with Ollama"
        )
        
        if selected_model != st.session_state.ollama_model:
            st.session_state.ollama_model = selected_model
            # Reset conversation if model changed and PDF already processed
            if st.session_state.pdf_processed:
                st.sidebar.warning("Model changed! Please re-process the PDF to use the new model.")
                st.session_state.pdf_processed = False
        
        st.sidebar.info(f"Using {selected_model} model via Ollama. Make sure Ollama is running.")
        ollama_url = st.sidebar.text_input("Ollama URL", value=OLLAMA_HOST)
        if ollama_url != OLLAMA_HOST:
            os.environ["OLLAMA_HOST"] = ollama_url
    
    # Advanced settings (collapsible)
    with st.sidebar.expander("Advanced Settings"):
        new_chunk_size = st.slider("Chunk Size", 500, 3000, st.session_state.current_chunk_size, 100, 
                                 help="Size of text chunks (larger chunks provide more context, smaller chunks more precision)")
        new_chunk_overlap = st.slider("Chunk Overlap", 50, 600, st.session_state.current_chunk_overlap, 50,
                                    help="Overlap between chunks to maintain context")
        new_retrieval_k = st.slider("Retrieval Count", 2, 10, st.session_state.current_retrieval_k, 1,
                                  help="Number of chunks to retrieve for each question")
        
        # Update settings if changed
        if (new_chunk_size != st.session_state.current_chunk_size or 
            new_chunk_overlap != st.session_state.current_chunk_overlap or 
            new_retrieval_k != st.session_state.current_retrieval_k):
            
            if st.button("Apply Settings"):
                # Update session state settings
                st.session_state.current_chunk_size = new_chunk_size
                st.session_state.current_chunk_overlap = new_chunk_overlap
                st.session_state.current_retrieval_k = new_retrieval_k
                
                # Mark for reprocessing
                if st.session_state.pdf_processed:
                    st.sidebar.warning("Settings changed! Please re-process the PDF.")
                    st.session_state.pdf_processed = False
    
    # File information
    st.sidebar.subheader("Current Document")
    st.sidebar.markdown(f"**Path:** {os.path.basename(PDF_PATH)}")
    
    # Process PDF button
    if not st.session_state.pdf_processed:
        if st.sidebar.button("📄 Process PDF Document", key="process_pdf"):
            if not st.session_state.use_free_model and GOOGLE_API_KEY == "your-google-api-key-here":
                st.sidebar.error("Please update the GOOGLE_API_KEY in the .env file.")
            elif PDF_PATH == "path/to/your/document.pdf":
                st.sidebar.error("Please update the PDF_PATH in the .env file.")
            else:
                process_pdf()
    else:
        st.sidebar.success("✅ PDF processed successfully")
        if st.sidebar.button("🔄 Re-Process PDF", key="reprocess_pdf"):
            process_pdf()
    
            # Document statistics
    if st.session_state.pdf_processed:
        st.sidebar.subheader("Document Statistics")
        stats = st.session_state.pdf_stats
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Pages", stats["pages"])
            st.metric("Characters", f"{stats['chars']:,}")
        
        with col2:
            st.metric("Text chunks", stats["chunks"])
            avg_chunk_size = stats["chars"] / stats["chunks"] if stats["chunks"] > 0 else 0
            st.metric("Avg. chunk size", f"{avg_chunk_size:.0f} chars")
        
        # Model information
        st.sidebar.subheader("Model Information")
        st.sidebar.info(f"Using: {st.session_state.model_used}")
        
        # Show average response time if available
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.sidebar.metric("Avg. response time", f"{avg_time:.2f}s")
        
        # Preview content
        with st.sidebar.expander("Preview Document Content"):
            if os.path.exists(PDF_PATH):
                raw_text, _ = extract_text_from_pdf(PDF_PATH)
                preview = raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text
                st.text_area("First 1000 characters", preview, height=200)
    
    # Main content area
    if st.session_state.pdf_processed:
        # Chat interface container
        st.markdown("### Ask me about the document")
        # More elegant input for questions
        user_question = st.text_input("", placeholder="Type your question here and press Enter...", key="question_input")
        
        if user_question:
            with st.spinner("Thinking..."):
                answer, source_docs, response_time = process_user_question(user_question)
            
            # Create a container for the chat history
            chat_container = st.container()
            
            # Display chat history in reverse order (newest at the bottom)
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:  # User message
                        st.markdown(f"""
                        <div class="user-message">
                            <strong>You:</strong> {html.escape(message.content)}
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # AI response
                        # For AI responses, we'll selectively allow HTML for tables and formatting
                        ai_content = message.content
                        # Check if there's any HTML in the content
                        if "<table" in ai_content.lower():
                            # It has a table, so we want to preserve that HTML
                            st.markdown(f"""
                            <div class="ai-message">
                                <strong>AI:</strong> {ai_content}
                                <div class="feedback-buttons">
                                    <button class="feedback-button">👍 Helpful</button>
                                    <button class="feedback-button">👎 Not helpful</button>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # No tables, so escape all HTML
                            st.markdown(f"""
                            <div class="ai-message">
                                <strong>AI:</strong> {html.escape(ai_content)}
                                <div class="feedback-buttons">
                                    <button class="feedback-button">👍 Helpful</button>
                                    <button class="feedback-button">👎 Not helpful</button>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show performance data
                st.caption(f"Response time: {response_time:.2f} seconds")
                
                # Show sources if available
                if source_docs and len(source_docs) > 0:
                    with st.expander("View Source Information"):
                        st.markdown(f"The AI used {len(source_docs)} sections from your document:")
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Source {i+1}:**")
                            preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                            st.text_area(f"Content snippet {i+1}", preview, height=100)
    else:
        # Instructions when PDF not processed
        st.info("👈 Please click 'Process PDF Document' in the sidebar to get started.")
        
        # Add some explanation of how the tool works
        st.markdown("""
        ## How this tool works
        
        1. **Process your PDF**: This tool extracts all text from your PDF and prepares it for question answering
        2. **Ask questions**: Once processed, you can ask questions about any content in the document
        3. **View sources**: For each answer, you can see which parts of the document were used
        
        
        ## Features
        
        - Uses advanced AI to provide accurate answers about your document content
        - Maintains conversation context for follow-up questions
        - Shows source information to verify answers
        - Now supports improved text processing and retrieval
        - Options to use powerful free models like Mistral, LLaMA 3, Mixtral, and more
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Enhanced PDF AI Assistant")
    model_name = st.session_state.model_used if hasattr(st.session_state, 'model_used') else "AI Model"
    st.sidebar.caption(f"Using {model_name}")

if __name__ == "__main__":
    # For command line usage
    parser = argparse.ArgumentParser(description="PDF Question Answering Tool")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--api_key", type=str, help="Google API Key")
    parser.add_argument("--use_free_model", action="store_true", help="Use free model via Ollama")
    parser.add_argument("--ollama_model", type=str, help="Ollama model to use (mistral, llama3, etc.)")
    
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    if args.pdf_path:
        PDF_PATH = args.pdf_path
    if args.api_key:
        GOOGLE_API_KEY = args.api_key
    if args.use_free_model:
        st.session_state.use_free_model = True
    if args.ollama_model:
        st.session_state.ollama_model = args.ollama_model
    
    main()