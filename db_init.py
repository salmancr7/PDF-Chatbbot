import os
import psycopg2  # PostgreSQL adapter for Python
from dotenv import load_dotenv  # Load environment variables

load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "work-demo")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "abcd123")

def init_db(recreate_tables=False):
    """Initialize the database with required tables.
    
    Args:
        recreate_tables: If True, drops existing tables before creating new ones
    """
    
    # Connection string
    conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Drop tables if requested (in reverse order of dependencies)
        if recreate_tables:
            print("Dropping existing tables...")
            cursor.execute("DROP TABLE IF EXISTS relevance_feedback CASCADE")
            cursor.execute("DROP TABLE IF EXISTS conversation_history CASCADE")
            cursor.execute("DROP TABLE IF EXISTS sessions CASCADE")
            print("Tables dropped successfully")
        
        # Create sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(36) PRIMARY KEY,
            pdf_path VARCHAR(255) NOT NULL,
            total_pages INTEGER DEFAULT 0,
            total_chunks INTEGER DEFAULT 0,
            embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
            llm_model VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create conversation_history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(36) REFERENCES sessions(session_id) ON DELETE CASCADE,
            role VARCHAR(10) NOT NULL,
            content TEXT NOT NULL,
            query_time FLOAT DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create relevance_feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS relevance_feedback (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER REFERENCES conversation_history(id) ON DELETE CASCADE,
            is_relevant BOOLEAN,
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    # Ask user if they want to recreate tables
    response = input("Do you want to recreate all tables? This will DELETE all existing data! (y/N): ")
    recreate = response.lower() == 'y'
    
    init_db(recreate_tables=recreate)
    print("Database initialization complete.")