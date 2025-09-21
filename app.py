import streamlit as st
import os
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import errorcode
import re

# Updated LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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

# Prompt template (works with both old and new LangChain)
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

def _create_chat_history_table_local():
    """Create chat_history table with mobile number field"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_timestamp DATETIME,
        user_name VARCHAR(255),
        user_email VARCHAR(255),
        user_mobile VARCHAR(20),
        user_question TEXT,
        assistant_answer LONGTEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()
        cur.execute(create_table_sql)
        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Error creating table: {err}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def _save_chat_entry_to_db_local(session_ts, user_name, user_email, user_mobile, question, assistant_answer):
    """Save chat entry to database with mobile number"""
    insert_sql = """
    INSERT INTO chat_history (session_timestamp, user_name, user_email, user_mobile, user_question, assistant_answer)
    VALUES (%s, %s, %s, %s, %s, %s)
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

        cur.execute(insert_sql, (session_ts_dt, user_name, user_email, user_mobile, question, assistant_answer))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
        conn.rollback()
        return False
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
    """Validate mobile number with international format support"""
    if not mobile:
        return False, "Mobile number is required"
    
    # Remove spaces, hyphens, and plus signs for validation
    clean_mobile = re.sub(r'[\s\-\+\(\)]', '', mobile)
    
    # Check if it contains only digits
    if not clean_mobile.isdigit():
        return False, "Mobile number should contain only digits, spaces, hyphens, or + sign"
    
    # Check length (international mobile numbers are typically 7-15 digits)
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
        
        # Validate required columns
        if 'prompt' not in df.columns or 'response' not in df.columns:
            st.error("Dataset must have 'prompt' and 'response' columns.")
            return None
            
        # Remove empty rows
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
        
        # Use updated embeddings if available, fallback to old version
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if not os.path.exists(VECTOR_DB_PATH):
            st.warning("Knowledgebase not found. Creating from dataset...")
            vectorstore = create_vector_db()
            return vectorstore
            
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def get_qa_chain(llm):
    """Create QA chain using available LangChain components"""
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
        
        if USE_NEW_LANGCHAIN:
            # Use new LangChain approach (no deprecation warnings)
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            return retrieval_chain
        else:
            # Fallback to old approach
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
    """Determine which LLM to use based on question content"""
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
            except Exception as fallback_e:
                st.error(f"Fallback to Gemini also failed: {fallback_e}")
                
    return None, llm_choice

# ============================================
# STREAMLIT APP
# ============================================
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "chat_history": [],
        "user_info_collected": False,
        "user_name": "",
        "user_email": "",
        "user_mobile": "",
        "session_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        st.info("Railway MySQL environment variables are required for chat history.")

    # User information collection with mobile number
    if not st.session_state.user_info_collected:
        st.subheader("👋 Welcome! Please provide your details to get started.")
        
        with st.form("user_info_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    "Full Name*", 
                    placeholder="Enter your full name",
                    help="Your full name for personalized assistance"
                )
                email = st.text_input(
                    "Email Address*", 
                    placeholder="your.email@example.com",
                    help="We'll use this for follow-ups if needed"
                )
            
            with col2:
                mobile = st.text_input(
                    "Mobile Number*", 
                    placeholder="+1234567890 or 1234567890",
                    help="Include country code for international numbers"
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
                    # All validations passed
                    st.session_state.user_name = name.strip()
                    st.session_state.user_email = email.strip().lower()
                    st.session_state.user_mobile = mobile.strip()
                    st.session_state.user_info_collected = True
                    st.success("✅ Welcome aboard! You can now start chatting.")
                    st.rerun()

    # Main chat interface
    if st.session_state.user_info_collected:
        # Display user info in sidebar
        with st.sidebar:
            st.subheader("👤 User Information")
            st.write(f"**Name:** {st.session_state.user_name}")
            st.write(f"**Email:** {st.session_state.user_email}")
            st.write(f"**Mobile:** {st.session_state.user_mobile}")
            st.write(f"**Session:** {st.session_state.session_timestamp}")
            
            if st.button("🔄 Reset Session", help="Clear chat history and start over"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()
        
        st.success(f"👋 Welcome, {st.session_state.user_name}!")
        
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
                            response = qa_chain.invoke({"input" if USE_NEW_LANGCHAIN else "query": question})
                            answer = response.get("answer" if USE_NEW_LANGCHAIN else "result", "I couldn't generate a proper response.")
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
                        st.session_state.user_name,
                        st.session_state.user_email,
                        st.session_state.user_mobile,  # Mobile number included
                        question,
                        answer
                    )
                    if saved:
                        st.success("💾 Chat saved to database", icon="✅")
                    else:
                        st.warning("⚠️ Failed to save chat to database.")
                except Exception as e:
                    st.error(f"Database error: {e}")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for sender, text in st.session_state.chat_history:
                with st.chat_message(sender):
                    st.markdown(text)

# ============================================
# INITIAL SETUP
# ============================================
def main():
    """Main function to run the app"""
    try:
        # Initialize database if configured
        if all([MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD]):
            success = create_chat_history_table()
            if success:
                print("✅ Database table initialized successfully")
            else:
                print("⚠️ Database table initialization failed")
        
        # Check for dataset
        if not os.path.exists(DATASET_PATH):
            print(f"Warning: {DATASET_PATH} not found. Create this file with 'prompt' and 'response' columns.")
        
        # Run the app
        run_app()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
