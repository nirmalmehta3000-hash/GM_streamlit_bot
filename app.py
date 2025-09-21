import streamlit as st
import os
import pandas as pd
from datetime import datetime
import mysql.connector
from mysql.connector import errorcode

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

# Database credentials
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
    if not (MYSQL_HOST and MYSQL_DATABASE and MYSQL_USER and MYSQL_PASSWORD):
        raise EnvironmentError("MySQL credentials are missing. Please set MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER and MYSQL_PASSWORD environment variables.")
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        autocommit=False
    )

def _create_chat_history_table_local():
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_timestamp DATETIME,
        user_name VARCHAR(255),
        user_email VARCHAR(255),
        user_question TEXT,
        assistant_answer LONGTEXT,
        user_mobile VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB] Error creating chat_history table: {e}")
        return False
    finally:
        if conn:
            conn.close()

def _save_chat_entry_to_db_local(session_ts, user_name, user_email, question, assistant_answer, user_mobile):
    insert_sql = """
    INSERT INTO chat_history (session_timestamp, user_name, user_email, user_question, assistant_answer, user_mobile)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        if isinstance(session_ts, str):
            try:
                session_ts_dt = datetime.strptime(session_ts, "%Y-%m-%d %H:%M:%S")
            except Exception:
                session_ts_dt = datetime.now()
        else:
            session_ts_dt = session_ts

        cur.execute(insert_sql, (session_ts_dt, user_name, user_email, question, assistant_answer, user_mobile))
        if cur.rowcount != 1:
            raise Exception(f"Insert affected {cur.rowcount} rows")
        conn.commit()
        cur.close()
        return True
    except mysql.connector.Error as db_err:
        if conn:
            conn.rollback()
        print(f"[DB] MySQL error: {db_err}")
        return False
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB] Error inserting chat entry: {e}")
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
# VECTORSTORE / QA
# ============================================
def create_vector_db():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at {DATASET_PATH}. Please upload it.")
        return None
    try:
        df = pd.read_excel(DATASET_PATH)
    except Exception as e:
        st.error(f"Error reading dataset file: {e}")
        return None

    documents = [Document(page_content=f"Q: {row.get('prompt','')}\nA: {row.get('response','')}") for _, row in df.iterrows()]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    st.success("Knowledgebase created and saved!")
    return vectorstore

def get_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(VECTOR_DB_PATH):
        st.warning("Knowledgebase not found. Creating from dataset...")
        vectorstore = create_vector_db()
        if vectorstore is None:
            return None
        return vectorstore
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def get_qa_chain(llm):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Could not load or create knowledgebase.")
        return None
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

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

def determine_llm(question: str):
    grok_keywords = ["grok", "xai", "x-ai", "grok model", "grok chat"]
    if any(word in question.lower() for word in grok_keywords) and ChatXAI and GROK_API_KEY:
        return "grok"
    return "gemini"

# ============================================
# STREAMLIT APP
# ============================================
def run_app():
    st.set_page_config(page_title="Gerryson Mehta Multi-LLM Chatbot", page_icon="🤖")
    st.title("Gerryson Mehta's Chatbot 🤖")
    st.write("Powered by Analytix Leap")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_info_collected" not in st.session_state:
        st.session_state.user_info_collected = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "user_mobile" not in st.session_state:
        st.session_state.user_mobile = ""
    if "session_timestamp" not in st.session_state:
        st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not (MYSQL_HOST and MYSQL_DATABASE and MYSQL_USER and MYSQL_PASSWORD):
        st.warning("MySQL environment variables are not fully set. Chats will NOT be saved until MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER and MYSQL_PASSWORD are configured.")

    if not st.session_state.user_info_collected:
        with st.form("user_info_form"):
            st.write("Please provide your details to start the chat.")
            name = st.text_input("Name")
            email = st.text_input("Email")
            mobile = st.text_input("Mobile Number")
            submit_button = st.form_submit_button("Start Chat")

            if submit_button:
                if not mobile.isdigit() or len(mobile) < 10:
                    st.error("❌ Please enter a valid mobile number")
                else:
                    st.session_state.user_name = name
                    st.session_state.user_email = email
                    st.session_state.user_mobile = mobile
                    st.session_state.user_info_collected = True
                    st.rerun()

    if st.session_state.user_info_collected:
        st.write(f"Welcome, {st.session_state.user_name}!")
        question = st.chat_input("Ask your career or customer service question here:")

        if question:
            llm_choice = determine_llm(question)

            llm = None
            if llm_choice == "gemini":
                if not GOOGLE_API_KEY:
                    st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
                else:
                    try:
                        llm = GoogleGenerativeAI(
                            model="gemini-1.5-flash-latest",
                            temperature=0.3,
                            google_api_key=GOOGLE_API_KEY,
                        )
                    except Exception as e:
                        st.error(f"Error initializing Gemini model: {e}.")
                        print(f"Error initializing Gemini model: {e}")
            elif llm_choice == "grok" and ChatXAI and GROK_API_KEY:
                try:
                    llm = ChatXAI(
                        model="grok-4",
                        temperature=0.3,
                        xai_api_key=GROK_API_KEY,
                    )
                except Exception as e:
                    st.error(f"Error initializing Grok model: {e}. Falling back to Gemini.")
                    print(f"Error initializing Grok model: {e}. Falling back to Gemini.")
                    llm_choice = "gemini"
                    if not GOOGLE_API_KEY:
                        st.error("Google API Key not found for fallback. Please set the GOOGLE_API_KEY environment variable.")
                    else:
                        try:
                            llm = GoogleGenerativeAI(
                                model="gemini-1.5-flash-latest",
                                temperature=0.3,
                                google_api_key=GOOGLE_API_KEY,
                            )
                        except Exception as gemini_e:
                            st.error(f"Error initializing fallback Gemini model: {gemini_e}.")
                            print(f"Error initializing fallback Gemini model: {gemini_e}")
                            llm = None
            else:
                if llm_choice == "grok":
                    st.warning("Grok model selected but langchain_xai is not available or API key is missing. Using Gemini instead.")
                llm_choice = "gemini"
                if not GOOGLE_API_KEY:
                    st.error("Google API Key not found for fallback. Please set the GOOGLE_API_KEY environment variable.")
                else:
                    try:
                        llm = GoogleGenerativeAI(
                            model="gemini-1.5-flash-latest",
                            temperature=0.3,
                            google_api_key=GOOGLE_API_KEY,
                        )
                    except Exception as e:
                        st.error(f"Error initializing fallback Gemini model: {e}.")
                        print(f"Error initializing fallback Gemini model: {e}")
                        llm = None

            answer = "Sorry, I could not get an answer from the model."
            if llm:
                qa_chain = get_qa_chain(llm)
                if qa_chain:
                    with st.spinner(f"Using {llm_choice.title()} model to answer..."):
                        try:
                            response = qa_chain.invoke({"query": question})
                            answer = response.get("result", "Could not get an answer from the model.")
                        except Exception as e:
                            answer = f"An error occurred while processing your request: {e}"
                            st.error(answer)
                            print(f"Error during QA chain invocation: {e}")
                else:
                    answer = "Sorry, I could not load the knowledge base to answer your question."
            else:
                answer = "Sorry, I encountered an issue with the language model and cannot answer your question at this time. Please ensure your API keys are correctly set."

            st.session_state.chat_history.append(("user", question))
            st.session_state.chat_history.append(("assistant", answer))

            try:
                saved = save_chat_entry_to_db(
                    st.session_state.session_timestamp,
                    st.session_state.user_name,
                    st.session_state.user_email,
                    question,
                    answer,
                    st.session_state.user_mobile
                )
                if saved:
                    st.info("Chat saved to database.")
                else:
                    st.warning("Failed to save chat to database. Check logs or DB credentials.")
            except EnvironmentError as env_err:
                st.warning(f"Database not configured: {env_err}")
            except Exception as e:
                st.error(f"Unexpected error saving to DB: {e}")
                print(f"[DB] Unexpected error saving to DB: {e}")

        for sender, text in st.session_state.chat_history:
            with st.chat_message(sender):
                st.markdown(text)

# ============================================
# INITIAL SETUP
# ============================================
try:
    ok = create_chat_history_table()
    if not ok:
        print("Warning: Could not create chat_history table automatically.")
except EnvironmentError as env_e:
    print(f"MySQL env not configured: {env_e}")
except Exception as e:
    print(f"Error creating chat_history table: {e}")

print("Attempting to create vector database (if dataset exists)...")
try:
    if os.path.exists(DATASET_PATH):
        pass
    else:
        print(f"Dataset not found at {DATASET_PATH}. Please upload it.")
except Exception as e:
    print(f"Error reading dataset file: {e}")

run_app()
