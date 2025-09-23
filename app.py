import streamlit as st
import os
import pandas as pd
from datetime import datetime
import mysql.connector
import re
import uuid
import hashlib

# LangChain imports - using stable versions to avoid errors
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Try importing user db_utils; fallback to local functions
try:
    from db_utils import create_chat_history_table, save_chat_entry_to_db
except Exception:
    create_chat_history_table = None
    save_chat_entry_to_db = None

# Import Grok if available
try:
    from langchain_xai import ChatXAI
except ImportError:
    ChatXAI = None

# API Keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROK_API_KEY = os.environ.get("GROK_API_KEY")

# Database credentials (Railway format)
MYSQL_HOST = os.environ.get("MYSQLHOST")
MYSQL_DATABASE = os.environ.get("MYSQLDATABASE") 
MYSQL_USER = os.environ.get("MYSQLUSER")
MYSQL_PASSWORD = os.environ.get("MYSQLPASSWORD")
MYSQL_PORT = int(os.environ.get("MYSQLPORT", 3306))

VECTOR_DB_PATH = "vectorstore.faiss"
DATASET_PATH = "dataset.xlsx"

# System instruction
SYSTEM_INSTRUCTION = (
    "You are a friendly, insightful customer service assistant for www.gerrysonmehta.com. "
    "Your expertise is helping aspiring data analysts, students, and professionals with career advice, project ideas, interview prep, portfolio building and time management. "
    "Respond conversationally, with empathy, encouragement and actionable steps; like a human expert mentor. "
    "Always personalize your guidance, use clear language and be honest when uncertain. Never limit help to code debugging. "
    "Refer to Gerryson Mehta's philosophy and provide motivation as needed."
)

# Prompt template
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user's question.
Respond conversationally, as a human expert coach.
If you don't know the answer, just say so honestly and avoid guessing.
----------------
{context}
Question: {question}
Expert Answer:
"""

# ============================================
# DATABASE HELPERS
# ============================================
def get_db_connection():
    """Get database connection with proper error handling"""
    if not all([MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD]):
        return None
    
    try:
        return mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4',
            autocommit=False
        )
    except mysql.connector.Error as err:
        st.error(f"Database connection failed: {err}")
        return None

def create_users_table():
    """Create users table for storing user information"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(50) PRIMARY KEY,
        full_name VARCHAR(200) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        mobile VARCHAR(20),
        username VARCHAR(100) UNIQUE,
        password_hash VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        last_login TIMESTAMP NULL,
        is_active BOOLEAN DEFAULT TRUE,
        profile_data JSON,
        INDEX idx_email (email),
        INDEX idx_username (username),
        INDEX idx_mobile (mobile)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        print("✅ Users table created/verified successfully")
        return True
    except Exception as e:
        print(f"Error creating users table: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def create_user_sessions_table():
    """Create user_sessions table for tracking user login sessions"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS user_sessions (
        session_id VARCHAR(50) PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        session_end TIMESTAMP NULL,
        is_active BOOLEAN DEFAULT TRUE,
        ip_address VARCHAR(45),
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
        INDEX idx_user_id (user_id),
        INDEX idx_session_start (session_start)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        print("✅ User sessions table created/verified successfully")
        return True
    except Exception as e:
        print(f"Error creating user_sessions table: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def _create_chat_history_table_local():
    """Create chat_history table with user_id reference"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_timestamp DATETIME,
        user_id VARCHAR(50),
        user_name VARCHAR(255),
        user_email VARCHAR(255),
        user_mobile VARCHAR(20),
        user_question TEXT,
        assistant_answer LONGTEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
        INDEX idx_user_id (user_id),
        INDEX idx_session_timestamp (session_timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error creating chat_history table: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def create_or_get_user(full_name, email, mobile):
    """Create new user or get existing user by email"""
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cur = conn.cursor()
        
        # First, try to get existing user by email
        select_sql = "SELECT user_id, full_name, email, mobile, last_login FROM users WHERE email = %s"
        cur.execute(select_sql, (email,))
        user = cur.fetchone()
        
        if user:
            # Update last login and mobile if changed
            user_id = user[0]
            update_sql = """
            UPDATE users 
            SET last_login = NOW(), mobile = %s, updated_at = NOW()
            WHERE user_id = %s
            """
            cur.execute(update_sql, (mobile, user_id))
            conn.commit()
            
            cur.close()
            return {
                'user_id': user[0],
                'full_name': user[1],
                'email': user[2],
                'mobile': mobile,  # Use updated mobile
                'last_login': user[4],
                'is_new': False
            }
        else:
            # Create new user
            user_id = str(uuid.uuid4())
            insert_sql = """
            INSERT INTO users (user_id, full_name, email, mobile, last_login)
            VALUES (%s, %s, %s, %s, NOW())
            """
            cur.execute(insert_sql, (user_id, full_name, email, mobile))
            conn.commit()
            
            cur.close()
            return {
                'user_id': user_id,
                'full_name': full_name,
                'email': email,
                'mobile': mobile,
                'last_login': datetime.now(),
                'is_new': True
            }
            
    except Exception as e:
        print(f"Error in create_or_get_user: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def create_user_session(user_id):
    """Create a new user session record"""
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        cur = conn.cursor()
        session_id = str(uuid.uuid4())
        
        insert_sql = """
        INSERT INTO user_sessions (session_id, user_id)
        VALUES (%s, %s)
        """
        cur.execute(insert_sql, (session_id, user_id))
        conn.commit()
        cur.close()
        return session_id
        
    except Exception as e:
        print(f"Error creating user session: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def _save_chat_entry_to_db_local(session_ts, user_name, user_email, user_mobile, question, assistant_answer):
    """Save chat entry to database with user_id reference"""
    # Get user_id from email
    user_id = None
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM users WHERE email = %s", (user_email,))
            result = cur.fetchone()
            if result:
                user_id = result[0]
            cur.close()
        except Exception as e:
            print(f"Error getting user_id: {e}")
        finally:
            conn.close()
    
    # Insert chat entry
    insert_sql = """
    INSERT INTO chat_history (session_timestamp, user_id, user_name, user_email, user_mobile, user_question, assistant_answer)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()
        if isinstance(session_ts, str):
            try:
                session_ts_dt = datetime.strptime(session_ts, "%Y-%m-%d %H:%M:%S")
            except Exception:
                session_ts_dt = datetime.now()
        else:
            session_ts_dt = session_ts or datetime.now()

        cur.execute(insert_sql, (session_ts_dt, user_id, user_name, user_email, user_mobile, question, assistant_answer))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Database save error: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_user_chat_history(user_id, limit=50):
    """Get chat history for a specific user"""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        cur = conn.cursor()
        select_sql = """
        SELECT user_question, assistant_answer, created_at 
        FROM chat_history 
        WHERE user_id = %s 
        ORDER BY created_at DESC 
        LIMIT %s
        """
        cur.execute(select_sql, (user_id, limit))
        rows = cur.fetchall()
        cur.close()
        
        return [{
            'question': row[0],
            'answer': row[1],
            'timestamp': row[2]
        } for row in rows]
        
    except Exception as e:
        print(f"Error getting user chat history: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_user_stats(user_id):
    """Get user statistics"""
    conn = get_db_connection()
    if not conn:
        return {}
        
    try:
        cur = conn.cursor()
        
        # Get total chats
        cur.execute("SELECT COUNT(*) FROM chat_history WHERE user_id = %s", (user_id,))
        total_chats = cur.fetchone()[0]
        
        # Get first chat date
        cur.execute("SELECT MIN(created_at) FROM chat_history WHERE user_id = %s", (user_id,))
        first_chat = cur.fetchone()[0]
        
        # Get last chat date
        cur.execute("SELECT MAX(created_at) FROM chat_history WHERE user_id = %s", (user_id,))
        last_chat = cur.fetchone()[0]
        
        cur.close()
        
        return {
            'total_chats': total_chats,
            'first_chat': first_chat,
            'last_chat': last_chat
        }
        
    except Exception as e:
        print(f"Error getting user stats: {e}")
        return {}
    finally:
        if conn:
            conn.close()

# Use local functions if db_utils not provided
if create_chat_history_table is None:
    create_chat_history_table = _create_chat_history_table_local
if save_chat_entry_to_db is None:
    save_chat_entry_to_db = _save_chat_entry_to_db_local

# ============================================
# UTILITY FUNCTIONS
# ============================================
def validate_mobile_number(mobile):
    """Validate mobile number"""
    if not mobile:
        return False, "Mobile number is required"
    
    # Remove spaces, hyphens, and plus signs for validation
    clean_mobile = re.sub(r'[\s\-\+\(\)]', '', mobile)
    
    if not clean_mobile.isdigit():
        return False, "Mobile number should contain only digits, spaces, hyphens, or + sign"
    
    if len(clean_mobile) < 7 or len(clean_mobile) > 15:
        return False, "Mobile number should be between 7-15 digits"
    
    return True, "Valid"

def validate_email(email):
    """Validate email format"""
    if not email:
        return False, "Email is required"
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False, "Please enter a valid email address"
    
    return True, "Valid"

def validate_name(name):
    """Validate name"""
    if not name or len(name.strip()) < 2:
        return False, "Name must be at least 2 characters long"
    
    return True, "Valid"

# ============================================
# VECTORSTORE / QA
# ============================================
def create_vector_db():
    """Create vector database from Excel dataset"""
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at {DATASET_PATH}. Please upload the dataset.xlsx file to your project directory.")
        st.info("Create a dataset.xlsx file with columns 'prompt' and 'response' containing your Q&A data.")
        return None
        
    try:
        df = pd.read_excel(DATASET_PATH)
        
        if 'prompt' not in df.columns or 'response' not in df.columns:
            st.error("Dataset must have 'prompt' and 'response' columns.")
            return None
            
        df = df.dropna(subset=['prompt', 'response'])
        
        if len(df) == 0:
            st.error("No valid data found in dataset.")
            return None
            
    except Exception as e:
        st.error(f"Error reading dataset file: {e}")
        return None

    try:
        documents = [
            Document(page_content=f"Q: {row.get('prompt','')}\nA: {row.get('response','')}")
            for _, row in df.iterrows()
        ]
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        
        st.success(f"Knowledgebase created with {len(texts)} text chunks!")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

def get_vectorstore():
    """Get or create vectorstore"""
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if not os.path.exists(VECTOR_DB_PATH):
            st.warning("Knowledgebase not found. Creating from dataset...")
            vectorstore = create_vector_db()
            return vectorstore
            
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def get_qa_chain(llm):
    """Create QA chain using stable LangChain components"""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Could not load or create knowledgebase.")
        return None
        
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Use stable LangChain components
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
        
        return RetrievalQA(
            retriever=retriever,
            combine_documents_chain=stuff_chain,
            return_source_documents=False
        )
        
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

def determine_llm(question: str):
    """Determine which LLM to use"""
    grok_keywords = ["grok", "xai", "x-ai", "grok model", "grok chat"]
    if any(word in question.lower() for word in grok_keywords) and ChatXAI and GROK_API_KEY:
        return "grok"
    return "gemini"

def initialize_llm(llm_choice):
    """Initialize LLM with proper error handling"""
    try:
        if llm_choice == "gemini":
            if not GOOGLE_API_KEY:
                st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
                return None, llm_choice
                
            return GoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY,
            ), llm_choice
            
        elif llm_choice == "grok" and ChatXAI and GROK_API_KEY:
            return ChatXAI(
                model="grok-beta",
                temperature=0.3,
                xai_api_key=GROK_API_KEY,
            ), llm_choice
            
    except Exception as e:
        st.error(f"Error initializing {llm_choice} model: {e}")
        
        # Fallback to Gemini
        if llm_choice != "gemini" and GOOGLE_API_KEY:
            try:
                return GoogleGenerativeAI(
                    model="gemini-1.5-flash-latest",
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY,
                ), "gemini"
            except Exception:
                pass
                
    return None, llm_choice

# ============================================
# STREAMLIT APP
# ============================================
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "chat_history": [],
        "user_info_collected": False,
        "user_data": None,
        "session_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_session_id": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def run_app():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Gerryson Mehta Multi-LLM Chatbot", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Gerryson Mehta's AI Assistant")
    st.markdown("*Powered by Analytix Leap - Your Career Growth Partner*")
    
    # Initialize session state
    initialize_session_state()
    
    # Check database configuration
    db_configured = all([MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD])
    if not db_configured:
        st.warning("⚠️ Database not configured. Chat history will not be saved.")

    # User information collection
    if not st.session_state.user_info_collected:
        st.subheader("👋 Welcome! Please provide your details to get started.")
        
        with st.form("user_info_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    "Full Name*", 
                    placeholder="Enter your full name"
                )
                email = st.text_input(
                    "Email Address*", 
                    placeholder="your.email@example.com"
                )
            
            with col2:
                mobile = st.text_input(
                    "Mobile Number*", 
                    placeholder="+1234567890 or 1234567890"
                )
                st.markdown("*All fields are required")
            
            submit_button = st.form_submit_button("🚀 Start Chat", use_container_width=True)

            if submit_button:
                # Validate all inputs
                name_valid, name_msg = validate_name(name)
                email_valid, email_msg = validate_email(email)
                mobile_valid, mobile_msg = validate_mobile_number(mobile)
                
                if not name_valid:
                    st.error(f"❌ {name_msg}")
                elif not email_valid:
                    st.error(f"❌ {email_msg}")
                elif not mobile_valid:
                    st.error(f"❌ {mobile_msg}")
                else:
                    # Create or get user from database
                    if db_configured:
                        user_data = create_or_get_user(name.strip(), email.strip().lower(), mobile.strip())
                        if user_data:
                            st.session_state.user_data = user_data
                            st.session_state.user_info_collected = True
                            
                            # Create user session
                            session_id = create_user_session(user_data['user_id'])
                            st.session_state.user_session_id = session_id
                            
                            if user_data['is_new']:
                                st.success(f"✅ Welcome aboard, {user_data['full_name']}! Your account has been created.")
                            else:
                                st.success(f"✅ Welcome back, {user_data['full_name']}! Great to see you again.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to create/retrieve user account. Please try again.")
                    else:
                        # Fallback for non-DB mode
                        st.session_state.user_data = {
                            'full_name': name.strip(),
                            'email': email.strip().lower(),
                            'mobile': mobile.strip(),
                            'user_id': str(uuid.uuid4()),
                            'is_new': True
                        }
                        st.session_state.user_info_collected = True
                        st.success("✅ Welcome aboard! You can now start chatting.")
                        st.rerun()

    # Main chat interface
    if st.session_state.user_info_collected and st.session_state.user_data:
        user_data = st.session_state.user_data
        
        # Display user info and stats in sidebar
        with st.sidebar:
            st.subheader("👤 User Information")
            st.write(f"**Name:** {user_data['full_name']}")
            st.write(f"**Email:** {user_data['email']}")
            st.write(f"**Mobile:** {user_data['mobile']}")
            st.write(f"**Session:** {st.session_state.session_timestamp}")
            
            if db_configured:
                # Show user stats
                user_stats = get_user_stats(user_data['user_id'])
                if user_stats:
                    st.subheader("📊 Your Stats")
                    st.metric("Total Chats", user_stats.get('total_chats', 0))
                    if user_stats.get('first_chat'):
                        st.write(f"**Member Since:** {user_stats['first_chat'].strftime('%Y-%m-%d')}")
                
                # Show recent chat history
                if st.button("📚 View Chat History"):
                    chat_history = get_user_chat_history(user_data['user_id'], 10)
                    if chat_history:
                        st.subheader("Recent Chats")
                        for chat in chat_history:
                            with st.expander(f"Chat from {chat['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                                st.write(f"**Q:** {chat['question'][:100]}...")
                                st.write(f"**A:** {chat['answer'][:200]}...")
                    else:
                        st.info("No previous chats found.")
        
        welcome_msg = f"👋 Welcome back, {user_data['full_name']}!" if not user_data.get('is_new') else f"👋 Welcome, {user_data['full_name']}!"
        st.success(welcome_msg)
        
        # Chat input
        question = st.chat_input("💬 Ask your career or technical question here...")

        if question:
            # Add user message to chat history immediately
            st.session_state.chat_history.append(("user", question))
            
            # Determine and initialize LLM
            llm_choice = determine_llm(question)
            llm, actual_choice = initialize_llm(llm_choice)
            
            # Process question
            answer = "Sorry, I couldn't process your question at this time."
            
            if llm:
                qa_chain = get_qa_chain(llm)
                
                if qa_chain:
                    with st.spinner(f"🧠 Using {actual_choice.title()} to find the best answer..."):
                        try:
                            response = qa_chain.invoke({"query": question})
                            answer = response.get("result", "I couldn't generate a proper response.")
                        except Exception as e:
                            answer = f"An error occurred while processing: {str(e)}"
                            st.error(f"Processing error: {e}")
                else:
                    answer = "Knowledge base is not available. Please ensure dataset.xlsx is uploaded."
            else:
                answer = "AI model is not available. Please check API key configuration."

            # Add assistant response to chat history
            st.session_state.chat_history.append(("assistant", answer))

            # Save to database if configured
            if db_configured:
                try:
                    saved = save_chat_entry_to_db(
                        st.session_state.session_timestamp,
                        user_data['full_name'],
                        user_data['email'],
                        user_data['mobile'],
                        question,
                        answer
                    )
                    if saved:
                        st.success("💾 Chat saved successfully!", icon="✅")
                    else:
                        st.warning("⚠️ Failed to save chat to database.")
                except Exception as e:
                    st.error(f"Database error: {e}")
                    print(f"Database save error: {e}")

        # Display chat history
        if st.session_state.chat_history:
            for sender, text in st.session_state.chat_history:
                with st.chat_message(sender):
                    st.markdown(text)

# ============================================
# INITIAL SETUP
# ============================================
def main():
    """Main function to run the app"""
    try:
        # Initialize database tables if configured
        if all([MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD]):
            # Create users table
            users_created = create_users_table()
            if users_created:
                print("✅ Users table initialized successfully")
            
            # Create user sessions table
            sessions_created = create_user_sessions_table()
            if sessions_created:
                print("✅ User sessions table initialized successfully")
            
            # Create chat history table (updated with user_id reference)
            chat_created = create_chat_history_table()
            if chat_created:
                print("✅ Chat history table initialized successfully")
            
            if not (users_created and sessions_created and chat_created):
                print("⚠️ Some database tables failed to initialize")
        
        # Check for dataset
        if not os.path.exists(DATASET_PATH):
            print(f"Warning: {DATASET_PATH} not found. Upload this file with 'prompt' and 'response' columns.")
        
        # Run the app
        run_app()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
