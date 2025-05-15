import os
import uuid
import argparse
import streamlit as st
import html
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage  # Add this line
from langchain_ollama import OllamaLLM
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
import re
import time
import pandas as pd
import io
# Import for Word documents
import docx2txt

# Load environment variables
load_dotenv()

# Configuration variables
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", r"C:\Users\PRECISION 7770\Downloads\document.pdf")
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

# Function to get database connection
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    return conn

# Set page configuration
st.set_page_config(
    page_title="Document AI Assistant", page_icon="📚", layout="wide", initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #1e1e2e; color: white; }
    .stButton button { background-color: #4CAF50; color: white; border-radius: 8px; border: none; padding: 10px 24px; font-size: 16px; }
    .stButton button:hover { background-color: #45a049; }
    .user-message { background-color: #2C3E50; color: white; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
    .ai-message { background-color: #3498DB; color: white; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
    .metrics-card { background-color: #2C3E50; color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 20px; margin-bottom: 20px; }
    .css-1v3fvcr { background-color: #1e1e2e; }
    .stTextInput input { background-color: #2C3E50; color: white; border: 1px solid #3498DB; }
    .streamlit-expanderHeader { background-color: #2C3E50; color: white; }
    .stMarkdown, h1, h2, h3, p, span { color: white; }
    .ai-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    .ai-table th, .ai-table td { border: 1px solid #4a6572; padding: 8px 12px; text-align: left; }
    .ai-table th { background-color: #2C3E50; color: white; }
    .ai-table tr { background-color: #34495e; color: white; }
    .ai-table tr:nth-child(even) { background-color: #3d566e; }
    .feedback-buttons { display: flex; justify-content: flex-end; margin-top: 5px; }
    .feedback-button { background-color: transparent; color: white; border: 1px solid #3498DB; padding: 5px 10px; margin-left: 5px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"chars": 0, "chunks": 0, "pages": 0}
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "use_free_model" not in st.session_state:
    st.session_state.use_free_model = USE_FREE_MODEL
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = OLLAMA_MODEL
if "response_times" not in st.session_state:
    st.session_state.response_times = []
if "current_chunk_size" not in st.session_state:
    st.session_state.current_chunk_size = CHUNK_SIZE
if "current_chunk_overlap" not in st.session_state:
    st.session_state.current_chunk_overlap = CHUNK_OVERLAP
if "current_retrieval_k" not in st.session_state:
    st.session_state.current_retrieval_k = RETRIEVAL_K
if "document_type" not in st.session_state:
    st.session_state.document_type = "pdf"

def extract_text_from_document(doc_path):
    """Extract text from PDF, Word document, or Excel file"""
    file_ext = os.path.splitext(doc_path)[1].lower()
    text = ""
    total_pages = 0
    
    try:
        if file_ext == '.pdf':
            # Extract from PDF
            pdf_reader = PdfReader(doc_path)
            total_pages = len(pdf_reader.pages)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            st.session_state.document_type = "pdf"
                    
        elif file_ext in ['.docx', '.doc']:
            # Extract from Word document
            text = docx2txt.process(doc_path)
            # Estimate pages (approx 3000 chars per page)
            total_pages = max(1, len(text) // 3000)
            st.session_state.document_type = "word"
            
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
            st.session_state.document_type = "excel"
            
        else:
            return None, 0, "Unsupported file format"
            
        if not text.strip():
            print(f"Warning: No text extracted from {file_ext} file.")
            
        return text, total_pages, st.session_state.document_type
        
    except Exception as e:
        print(f"Error reading document: {str(e)}")
        return None, 0, None

def clean_text(text):
    """Clean extracted text to improve quality"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    return text

def split_text_into_chunks(text):
    """Split text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.current_chunk_size,
        chunk_overlap=st.session_state.current_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    if chunks:
        avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
        print(f"Created {len(chunks)} chunks with average size of {avg_chunk_size:.1f} characters")
    
    return chunks

def create_vector_store(text_chunks):
    """Create vector store with text embeddings"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def create_conversation_chain(vector_store):
    """Create an enhanced conversation chain for comprehensive Q&A"""
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": st.session_state.current_retrieval_k}
    )
    
    if st.session_state.use_free_model:
        try:
            model_to_use = ADVANCED_MODEL if ADVANCED_MODEL and ADVANCED_MODEL != "mistral" else st.session_state.ollama_model
            
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
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )
        st.session_state.model_used = "Google Gemini"
    
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
    If creating tables, use proper HTML table tags with the "ai-table" class.
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
        verbose=True
    )
    
    return conversation_chain

def sanitize_ai_response(response_text):
    """Clean AI response to prevent HTML rendering issues"""
    response_text = re.sub(r'</div>', '', response_text)
    
    if "<table" in response_text.lower():
        response_text = re.sub(r'<table', '<table class="ai-table"', response_text, flags=re.IGNORECASE)
        
        if response_text.count("<table") > response_text.count("</table"):
            response_text += "</table>"
    
    common_tags = ["div", "p", "span", "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6"]
    for tag in common_tags:
        opening_count = len(re.findall(f"<{tag}[^>]*>", response_text, re.IGNORECASE))
        closing_count = len(re.findall(f"</{tag}>", response_text, re.IGNORECASE))
        
        if opening_count > closing_count:
            for _ in range(opening_count - closing_count):
                response_text += f"</{tag}>"
    
    return response_text

def process_user_question(user_question):
    """Process user question with improved accuracy and response tracking"""
    if st.session_state.conversation is None:
        st.error("Please process the document first.")
        return
    
    start_time = time.time()
    
    if st.session_state.session_id:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content FROM conversation_history WHERE session_id = %s ORDER BY timestamp",
                (st.session_state.session_id,)
            )
            
            messages = []
            for role, content in cursor.fetchall():
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            
            if messages and hasattr(st.session_state.conversation, 'memory'):
                st.session_state.conversation.memory.chat_memory.messages = messages
        except Exception as e:
            st.error(f"Error loading conversation history: {str(e)}")
        finally:
            conn.close()
    
    response = st.session_state.conversation({"question": user_question})
    
    response_time = time.time() - start_time
    st.session_state.response_times.append(response_time)
    
    answer_text = sanitize_ai_response(response["answer"])
    
    if st.session_state.session_id:
        conn = get_db_connection()
        try:
            conn.autocommit = False
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (%s, %s, %s)",
                (st.session_state.session_id, "human", user_question)
            )
            
            cursor.execute(
                "INSERT INTO conversation_history (session_id, role, content, query_time) VALUES (%s, %s, %s, %s)",
                (st.session_state.session_id, "ai", answer_text, response_time)
            )
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            st.error(f"Error storing conversation: {str(e)}")
        finally:
            cursor.close()
            conn.close()
    
    history = response["chat_history"]
    if len(history) > 0 and isinstance(history[-1], AIMessage):
        history[-1] = AIMessage(content=answer_text)
    
    st.session_state.chat_history = history
    
    return answer_text, response.get("source_documents", []), response_time

def process_document():
    """Process the document with improved text extraction and chunking"""
    if not os.path.exists(DOCUMENT_PATH):
        st.error(f"Document file not found at: {DOCUMENT_PATH}")
        return False
    
    with st.spinner("Processing document - this may take a while for large files..."):
        progress_text = st.empty()
        progress_text.text("Extracting text from document...")
        
        raw_text, total_pages, doc_type = extract_text_from_document(DOCUMENT_PATH)
        
        if not raw_text:
            st.error("Failed to extract text from the document. The file might be protected or in an unsupported format.")
            return False
        
        progress_bar = st.progress(0)
        
        progress_text.text("Cleaning and preparing text...")
        progress_bar.progress(20)
        cleaned_text = clean_text(raw_text)
        
        progress_text.text(f"Extracted {len(cleaned_text)} characters of text from {total_pages} {'pages' if doc_type != 'excel' else 'sheets'}")
        
        progress_text.text("Splitting text into optimal chunks...")
        progress_bar.progress(40)
        text_chunks = split_text_into_chunks(cleaned_text)
        
        progress_text.text("Creating vector database with embeddings...")
        progress_bar.progress(60)
        st.session_state.vector_store = create_vector_store(text_chunks)
        
        progress_text.text("Setting up Q&A system with enhanced prompts...")
        progress_bar.progress(80)
        st.session_state.conversation = create_conversation_chain(st.session_state.vector_store)
        
        # Create session in database
        session_id = str(uuid.uuid4())
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            # Check if the table exists and if needed columns exist
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
                    (session_id, DOCUMENT_PATH, total_pages, len(text_chunks), st.session_state.model_used)
                )
            else:
                cursor.execute(
                    """INSERT INTO sessions 
                       (session_id, pdf_path, llm_model) 
                       VALUES (%s, %s, %s)""",
                    (session_id, DOCUMENT_PATH, st.session_state.model_used)
                )
            conn.commit()
            st.session_state.session_id = session_id
        except Exception as e:
            st.error(f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        progress_bar.progress(100)
        progress_text.text("Document processing complete!")
        st.session_state.doc_processed = True
        
        # Save stats
        st.session_state.doc_stats = {
            "chars": len(cleaned_text),
            "chunks": len(text_chunks),
            "pages": total_pages,
            "doc_type": doc_type,
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
        st.title("Document AI Assistant")
        st.markdown("Ask detailed questions about your documents (PDF, Word, Excel) and get accurate AI-powered answers.")
    
    # Sidebar for configuration and controls
    st.sidebar.header("📋 Document Control")
    
    # Model selection options
    st.session_state.use_free_model = st.sidebar.checkbox(
        "Use free local model",
        value=st.session_state.use_free_model,
        help="Use locally hosted models via Ollama instead of Google Gemini"
    )
    
    if st.session_state.use_free_model:
        available_models = ["mistral", "llama3", "mixtral", "phi3", "gemma"]
        selected_model = st.sidebar.selectbox(
            "Select Ollama Model",
            available_models,
            index=available_models.index(st.session_state.ollama_model) if st.session_state.ollama_model in available_models else 0,
            help="Choose which model to use with Ollama"
        )
        
        if selected_model != st.session_state.ollama_model:
            st.session_state.ollama_model = selected_model
            if st.session_state.doc_processed:
                st.sidebar.warning("Model changed! Please re-process the document to use the new model.")
                st.session_state.doc_processed = False
        
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
        
        if (new_chunk_size != st.session_state.current_chunk_size or 
            new_chunk_overlap != st.session_state.current_chunk_overlap or 
            new_retrieval_k != st.session_state.current_retrieval_k):
            
            if st.button("Apply Settings"):
                st.session_state.current_chunk_size = new_chunk_size
                st.session_state.current_chunk_overlap = new_chunk_overlap
                st.session_state.current_retrieval_k = new_retrieval_k
                
                if st.session_state.doc_processed:
                    st.sidebar.warning("Settings changed! Please re-process the document.")
                    st.session_state.doc_processed = False
    
    # File information
    st.sidebar.subheader("Current Document")
    st.sidebar.markdown(f"**Path:** {os.path.basename(DOCUMENT_PATH)}")
    
    # Process document button
    if not st.session_state.doc_processed:
        if st.sidebar.button("📄 Process Document", key="process_doc"):
            if not st.session_state.use_free_model and GOOGLE_API_KEY == "your-google-api-key-here":
                st.sidebar.error("Please update the GOOGLE_API_KEY in the .env file.")
            elif DOCUMENT_PATH == "path/to/your/document.pdf":
                st.sidebar.error("Please update the DOCUMENT_PATH in the .env file.")
            else:
                process_document()
    else:
        st.sidebar.success("✅ Document processed successfully")
        if st.sidebar.button("🔄 Re-Process Document", key="reprocess_doc"):
            process_document()
    
    # Document statistics
    if st.session_state.doc_processed:
        st.sidebar.subheader("Document Statistics")
        stats = st.session_state.doc_stats
        doc_type = stats.get("doc_type", "document")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if doc_type == "excel":
                st.metric("Sheets", stats["pages"])
            else:
                st.metric("Pages", stats["pages"])
            st.metric("Characters", f"{stats['chars']:,}")
        
        with col2:
            st.metric("Text chunks", stats["chunks"])
            st.metric("Document type", doc_type.upper())
        
        # Model information
        st.sidebar.subheader("Model Information")
        st.sidebar.info(f"Using: {st.session_state.model_used}")
        
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.sidebar.metric("Avg. response time", f"{avg_time:.2f}s")
        
        # Preview content
        with st.sidebar.expander("Preview Document Content"):
            if os.path.exists(DOCUMENT_PATH):
                raw_text, _, _ = extract_text_from_document(DOCUMENT_PATH)
                preview = raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text
                st.text_area("First 1000 characters", preview, height=200)
    
    # Main content area
    if st.session_state.doc_processed:
        # Chat interface container
        st.markdown("### Ask me about the document")
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
        # Instructions when document not processed
        st.info("👈 Please click 'Process Document' in the sidebar to get started.")
        
        # Add some explanation of how the tool works
        st.markdown("""
        ## How this tool works
        
        1. **Process your document**: This tool extracts all text from your document (PDF, Word, or Excel) and prepares it for question answering
        2. **Ask questions**: Once processed, you can ask questions about any content in the document
        3. **View sources**: For each answer, you can see which parts of the document were used
        
        ## Supported file formats
        
        - **PDF files** (.pdf): Text-based PDF documents
        - **Word documents** (.docx, .doc): Microsoft Word files
        - **Excel spreadsheets** (.xlsx, .xls): Microsoft Excel files
        - **CSV files** (.csv): Comma-separated values
        
        ## Features
        
        - Uses advanced AI to provide accurate answers about your document content
        - Maintains conversation context for follow-up questions
        - Shows source information to verify answers
        - Options to use powerful free models like Mistral, LLaMA 3, Mixtral, and more
        """)
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Document AI Assistant")
    model_name = st.session_state.model_used if hasattr(st.session_state, 'model_used') else "AI Model"
    st.sidebar.caption(f"Using {model_name}")

if __name__ == "__main__":
    # For command line usage
    parser = argparse.ArgumentParser(description="Document Question Answering Tool")
    parser.add_argument("--doc_path", type=str, help="Path to the document file (PDF, Word, Excel)")
    parser.add_argument("--api_key", type=str, help="Google API Key")
    parser.add_argument("--use_free_model", action="store_true", help="Use free model via Ollama")
    parser.add_argument("--ollama_model", type=str, help="Ollama model to use (mistral, llama3, etc.)")
    
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    if args.doc_path:
        DOCUMENT_PATH = args.doc_path
    if args.api_key:
        GOOGLE_API_KEY = args.api_key
    if args.use_free_model:
        st.session_state.use_free_model = True
    if args.ollama_model:
        st.session_state.ollama_model = args.ollama_model
    
    main()