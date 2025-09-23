import streamlit as st
import os
import pandas as pd
from datetime import datetime
import re
import uuid
import traceback

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

# Import database utilities - now using external db_utils.py
try:
    from db_utils import (
        initialize_all_tables,
        test_connection,
        create_or_get_user,
        create_user_session,
        save_chat_entry_to_db,
        get_user_chat_history,
        get_user_stats,
        get_database_info,
        get_db_connection
    )
    DB_UTILS_AVAILABLE = True
    print("✅ Successfully imported db_utils module")
except ImportError as e:
    DB_UTILS_AVAILABLE = False
    st.error(f"❌ Could not import db_utils: {e}")
    st.error("Please ensure db_utils.py is in the same directory as app.py")

# Import Grok if available
try:
    from langchain_xai import ChatXAI
except ImportError:
    ChatXAI = None

# API Keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROK_API_KEY = os.environ.get("GROK_API_KEY")

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
# DEBUG FUNCTIONS
# ============================================
def debug_database_connection():
    """Debug database connection and table structure"""
    if not DB_UTILS_AVAILABLE:
        return False, "db_utils not available"
    
    try:
        # Test basic connection
        conn = get_db_connection()
        if not conn:
            return False, "Could not establish database connection"
        
        # Test tables exist
        cur = conn.cursor()
        cur.execute("SHOW TABLES")
        tables = [row[0] for row in cur.fetchall()]
        
        required_tables = ['users', 'chat_history', 'user_sessions']
        missing_tables = [t for t in required_tables if t not in tables]
        
        if missing_tables:
            cur.close()
            conn.close()
            return False, f"Missing tables: {missing_tables}"
        
        # Test chat_history table structure
        cur.execute("DESCRIBE chat_history")
        columns = [row[0] for row in cur.fetchall()]
        
        required_columns = ['user_question', 'assistant_answer', 'user_id', 'user_name', 'user_email']
        missing_columns = [c for c in required_columns if c not in columns]
        
        cur.close()
        conn.close()
        
        if missing_columns:
            return False, f"Missing columns in chat_history: {missing_columns}"
        
        return True, "Database structure OK"
        
    except Exception as e:
        return False, f"Database debug error: {str(e)}"

def test_chat_save(user_data):
    """Test saving a chat entry"""
    if not DB_UTILS_AVAILABLE:
        return False, "db_utils not available"
    
    try:
        test_question = "Test question"
        test_answer = "Test answer"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = save_chat_entry_to_db(
            timestamp,
            user_data['full_name'],
            user_data['email'],
            user_data['mobile'],
            test_question,
            test_answer
        )
        
        return result, "Test completed"
        
    except Exception as e:
        return False, f"Test save error: {str(e)}"

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
        "user_session_id": None,
        "debug_mode": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_database_status():
    """Check database connection and display status"""
    if not DB_UTILS_AVAILABLE:
        return False
    
    try:
        if test_connection():
            st.sidebar.success("🟢 Database Connected")
            
            # Show database info
            db_info = get_database_info()
            if db_info:
                st.sidebar.text(f"Database: {db_info.get('database', 'N/A')}")
                st.sidebar.text(f"Tables: {db_info.get('total_tables', 0)}")
                
                # Show table counts
                if 'tables' in db_info:
                    for table, count in db_info['tables'].items():
                        st.sidebar.text(f"  {table}: {count} records")
            
            return True
        else:
            st.sidebar.error("🔴 Database Connection Failed")
            return False
    except Exception as e:
        st.sidebar.error(f"🔴 Database Error: {str(e)[:50]}...")
        return False

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
    
    # Add debug toggle
    if st.sidebar.checkbox("Debug Mode"):
        st.session_state.debug_mode = True
    
    # Check database status
    db_available = check_database_status()
    
    # Show debug info if enabled
    if st.session_state.debug_mode:
        st.sidebar.subheader("🔍 Debug Information")
        
        if DB_UTILS_AVAILABLE:
            db_ok, db_msg = debug_database_connection()
            if db_ok:
                st.sidebar.success(f"✅ {db_msg}")
            else:
                st.sidebar.error(f"❌ {db_msg}")
        else:
            st.sidebar.error("❌ db_utils module not available")
    
    if not db_available:
        st.warning("⚠️ Database not available. Chat history will not be saved.")

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
                    if db_available:
                        try:
                            user_data = create_or_get_user(name.strip(), email.strip().lower(), mobile.strip())
                            if user_data:
                                st.session_state.user_data = user_data
                                st.session_state.user_info_collected = True
                                
                                # Create user session
                                session_id = create_user_session(user_data['user_id'])
                                st.session_state.user_session_id = session_id
                                
                                # Test chat save in debug mode
                                if st.session_state.debug_mode:
                                    test_ok, test_msg = test_chat_save(user_data)
                                    if test_ok:
                                        st.success(f"✅ Database test passed: {test_msg}")
                                    else:
                                        st.error(f"❌ Database test failed: {test_msg}")
                                
                                if user_data['is_new']:
                                    st.success(f"✅ Welcome aboard, {user_data['full_name']}! Your account has been created.")
                                else:
                                    st.success(f"✅ Welcome back, {user_data['full_name']}! Great to see you again.")
                                st.rerun()
                            else:
                                st.error("❌ Failed to create/retrieve user account. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Database error during user creation: {str(e)}")
                            if st.session_state.debug_mode:
                                st.error(f"Full error: {traceback.format_exc()}")
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
            
            if db_available:
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

            # Save to database if available
            if db_available:
                try:
                    if st.session_state.debug_mode:
                        st.info(f"Attempting to save chat for user: {user_data['user_id']}")
                        st.info(f"Question length: {len(question)} chars")
                        st.info(f"Answer length: {len(answer)} chars")
                    
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
                        if st.session_state.debug_mode:
                            st.error("save_chat_entry_to_db returned False")
                            
                except Exception as e:
                    st.error(f"Database error: {str(e)}")
                    if st.session_state.debug_mode:
                        st.error(f"Full traceback: {traceback.format_exc()}")
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
        # Initialize database tables if db_utils is available
        if DB_UTILS_AVAILABLE:
            print("🔄 Initializing database tables...")
            if initialize_all_tables():
                print("✅ All database tables initialized successfully")
            else:
                print("⚠️ Some database tables failed to initialize")
        else:
            print("⚠️ db_utils module not available - running without database")
        
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
