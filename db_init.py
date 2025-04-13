import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "pdf_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")

def init_db():
    """Initialize the database with required tables."""
    
    # Connection string
    conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Create sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(36) PRIMARY KEY,
            pdf_path VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create conversation_history table with TEXT type for content
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(36) REFERENCES sessions(session_id) ON DELETE CASCADE,
            role VARCHAR(10) NOT NULL,
            content TEXT NOT NULL,  -- Using TEXT type to store large responses
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create index for faster querying
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_session 
        ON conversation_history(session_id)
        """)
        
        print("Database tables created successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    init_db()