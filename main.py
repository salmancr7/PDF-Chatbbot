import os
import uuid
import argparse
import streamlit as st
import html  # Added for HTML escaping
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.llms import Ollama  # Added for free model support
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import re  # Added for regex pattern matching

# Load environment variables
load_dotenv()

# Configuration variables
PDF_PATH = os.getenv("PDF_PATH", r"C:\Users\PRECISION 7770\Downloads\RFP for Integrated Labour Management System - ver 21.2.pdf")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_FREE_MODEL = os.getenv("USE_FREE_MODEL", "false").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "work-demo")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")

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
    page_title="PDF AI Assistant", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# PDF AI Assistant\nA powerful tool to interact with your PDF documents using AI."
    }
)

# Custom CSS for better UI (unchanged)
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
    /* Make input box dark themed */
    .stTextInput input {
        background-color: #2C3E50;
        color: white;
        border: 1px solid #3498DB;
    }
    /* Style expanders */
    .streamlit-expanderHeader {
        background-color: #2C3E50;
        color: white;
    }
    /* Change text color to white for better visibility on dark background */
    .stMarkdown, h1, h2, h3, p, span {
        color: white;
    }
    /* Style tables */
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
            
            # Optional progress indication for large PDFs
            if total_pages > 50 and i % 10 == 0:
                print(f"Processed {i}/{total_pages} pages...")
        
        if not text.strip():
            print("Warning: No text was extracted from the PDF. The file might be scanned images or protected.")
            
        return text, total_pages
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return None, 0

def split_text_into_chunks(text):
    """Split text into manageable chunks with appropriate overlap for context preservation"""
    # Increased chunk size substantially to ensure better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,       # Larger chunks for more context
        chunk_overlap=500,     # Substantial overlap to preserve context
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]  # Try to split at natural boundaries first
    )
    
    chunks = text_splitter.split_text(text)
    
    # Verify we're getting reasonable chunks
    if chunks:
        # Log some diagnostics about the chunks
        avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
        print(f"Created {len(chunks)} chunks with average size of {avg_chunk_size:.1f} characters")
    else:
        print("Warning: No chunks were created. Input text might be empty.")
    
    return chunks

def create_vector_store(text_chunks):
    """Create vector store with text embeddings"""
    try:
        # Use a free local embedding model instead of OpenAI
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Print chunk count for verification
        print(f"Creating vector store with {len(text_chunks)} chunks")
        
        # Create the vector store
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Verify vector store creation was successful
        if vector_store:
            print("Vector store created successfully")
        
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def create_conversation_chain(vector_store):
    """Create an enhanced conversation chain for comprehensive Q&A across the entire document"""
    
    # Set up a more effective retriever with higher top_k to get more context
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Use similarity search
        search_kwargs={
            "k": 7  # Increased to retrieve more chunks for better context
        }
    )
    
    # Choose between Google Gemini or Ollama (free, local model)
    if st.session_state.use_free_model:
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
            st.warning("Failed to connect to Ollama. Falling back to Google Gemini.")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
    else:
        # Use Google's Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # Updated model name
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
    
    # Configure memory with the explicit output key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",  # Specify which output to store in memory
        return_messages=True
    )
    
    # Configure custom prompt to encourage comprehensive answers with supplementary knowledge
    custom_template = """
    You are an AI assistant that answers questions based on PDF documents.
    You should provide thorough, comprehensive answers based on the content from the PDF document.
    
    Chat History:
    {chat_history}
    
    Context from the document:
    {context}
    
    Question: {question}
    
    Answer the question as completely as possible using the provided context.
    If the answer is not fully covered in the context, you can add supplementary information to
    make your answer more helpful. In these cases, clearly indicate what information comes
    from outside the document by prefacing with "Additionally:" or similar phrasing.
    
    When creating tables or structured content, format it in a clean way with proper HTML table tags.
    Do not include any closing </div> tags or other HTML tags that could disrupt the formatting.
    Always close any HTML tags you open.
    
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
        return_source_documents=True,  # Optional: returns source chunks for verification
        verbose=True  # Helpful for debugging
    )
    
    return conversation_chain

# Function to clean and sanitize AI responses
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
    """Process user question and get comprehensive response from the full document"""
    if st.session_state.conversation is None:
        st.error("Please process the PDF document first.")
        return
    
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
            
            # Store AI response - ensure we're not truncating the content
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (st.session_state.session_id, "ai", answer_text)
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
    
    return answer_text, response.get("source_documents", [])

def validate_processing(total_chars, total_chunks, total_pages):
    """Validate that the PDF processing was likely successful"""
    if total_chars == 0:
        return False, "No text was extracted from the PDF. The file might be protected or contain only images."
    
    # Calculate expected number of chunks based on doc size
    expected_min_chunks = max(1, total_chars // 5000)  # Very rough estimate
    
    if total_chunks < expected_min_chunks:
        return False, f"Warning: Fewer chunks ({total_chunks}) created than expected for document size. Processing may be incomplete."
    
    # Calculate average characters per page
    avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
    
    if avg_chars_per_page < 100 and total_pages > 1:
        return False, f"Warning: Low text content per page ({avg_chars_per_page:.1f} chars/page). Some content may not have been extracted."
    
    return True, "PDF processing appears successful."

def process_pdf():
    """Process the entire PDF file at the configured path"""
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
        
        # Show PDF content length information
        progress_text.text(f"Extracted {len(raw_text)} characters of text from {total_pages} pages")
        
        # Process the extracted text
        progress_text.text("Splitting text into chunks...")
        progress_bar.progress(30)
        text_chunks = split_text_into_chunks(raw_text)
        
        # Create vector store
        progress_text.text("Creating vector database...")
        progress_bar.progress(60)
        st.session_state.vector_store = create_vector_store(text_chunks)
        
        # Create conversation chain
        progress_text.text("Setting up Q&A system...")
        progress_bar.progress(90)
        st.session_state.conversation = create_conversation_chain(st.session_state.vector_store)
        
        # Validate processing
        success, message = validate_processing(len(raw_text), len(text_chunks), total_pages)
        
        if not success:
            st.warning(message)
        
        # Create session in database
        session_id = str(uuid.uuid4())
        conn = get_db_connection()
        try:
            conn.autocommit = False
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (session_id, pdf_path) VALUES (%s, %s)",
                (session_id, PDF_PATH)
            )
            conn.commit()
            st.session_state.session_id = session_id
        except Exception as e:
            conn.rollback()
            st.error(f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        progress_bar.progress(100)
        progress_text.text("PDF processing complete!")
        st.session_state.pdf_processed = True
        
        # Save stats
        st.session_state.pdf_stats = {
            "chars": len(raw_text),
            "chunks": len(text_chunks),
            "pages": total_pages
        }
        
        return True
    
    return False

def format_bytes(size):
    """Format bytes to a readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def main():
    # Main layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
    
    with col2:
        st.title("Advanced PDF AI Assistant")
        st.markdown("Ask questions about your PDF document and get AI-powered answers.")
    
    # Sidebar for configuration and controls
    st.sidebar.header("📋 Document Control")
    
    # New option for free model
    st.session_state.use_free_model = st.sidebar.checkbox(
        "Use free Mistral model (requires Ollama)",
        value=st.session_state.use_free_model,
        help="Use locally hosted Mistral model via Ollama instead of Google Gemini"
    )
    
    if st.session_state.use_free_model:
        st.sidebar.info("Using Mistral model via Ollama. Make sure Ollama is running.")
        ollama_url = st.sidebar.text_input("Ollama URL", value=OLLAMA_HOST)
        if ollama_url != OLLAMA_HOST:
            os.environ["OLLAMA_HOST"] = ollama_url
    
    # File information
    st.sidebar.subheader("Current Document")
    st.sidebar.markdown(f"**Path:** {os.path.basename(PDF_PATH)}")
    
    # Process PDF button with improved styling
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
                answer, source_docs = process_user_question(user_question)
            
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
                        # For AI responses, we'll selectively allow some HTML for tables and formatting
                        # This is safer than directly using message.content with unsafe_allow_html=True
                        ai_content = message.content
                        # Check if there's any HTML in the content
                        if "<table" in ai_content.lower():
                            # It has a table, so we want to preserve that HTML
                            st.markdown(f"""
                            <div class="ai-message">
                                <strong>AI:</strong> {ai_content}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # No tables, so escape all HTML
                            st.markdown(f"""
                            <div class="ai-message">
                                <strong>AI:</strong> {html.escape(ai_content)}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show sources if available
                if source_docs and len(source_docs) > 0:
                    with st.expander("View Source Information"):
                        st.markdown("The AI used the following sections from your document:")
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Source {i+1}:**")
                            preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                            st.text_area(f"Content snippet {i+1}", preview, height=100)
    else:
        # Instructions when PDF not processed
        st.info("👈 Please click 'Process PDF Document' in the sidebar to get started.")
        
        # Add some explanation of how the tool works
        st.markdown("""
        ## How this works
        
        1. **Process your PDF**: This tool extracts all text from your PDF and prepares it for question answering
        2. **Ask questions**: Once processed, you can ask questions about any content in the document
        3. **View sources**: For each answer, you can see which parts of the document were used
        
        ## Features
        
        - Uses advanced AI to provide accurate answers about your document content
        - Maintains conversation context for follow-up questions
        - Shows source information to verify answers
        - Now supports supplementary information beyond the document
        - Option to use free Mistral model for better privacy and no API costs
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("PDF AI Assistant v2.1")
    model_name = "Mistral (Free)" if st.session_state.use_free_model else "Google Gemini"
    st.sidebar.caption(f"Using LangChain + {model_name}")

if __name__ == "__main__":
    # For command line usage
    parser = argparse.ArgumentParser(description="PDF Question Answering Tool")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--api_key", type=str, help="Google API Key")
    parser.add_argument("--use_free_model", action="store_true", help="Use free Mistral model via Ollama")
    
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    if args.pdf_path:
        PDF_PATH = args.pdf_path
    if args.api_key:
        GOOGLE_API_KEY = args.api_key
    if args.use_free_model:
        st.session_state.use_free_model = True
    
    main()