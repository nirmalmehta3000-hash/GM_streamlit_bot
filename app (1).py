import streamlit as st
import os
import pandas as pd
from datetime import datetime
import csv # Import the csv module

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain # Import StuffDocumentsChain


# Import LLM models (if you intend to use Grok)
try:
    from langchain_xai import ChatXAI
except ImportError:
    ChatXAI = None # Handle case where langchain_xai is not installed

# Access API keys from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROK_API_KEY = os.environ.get("GROK_API_KEY")

VECTOR_DB_PATH = "vectorstore.faiss"
DATASET_PATH = "dataset.xlsx" # Make sure this file is uploaded in /content
CHAT_HISTORY_DIR = "chat_history"
CHAT_HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "chat_history.csv")


# Expert system instruction (shared)
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


def create_vector_db():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at {{DATASET_PATH}}. Please upload it.")
        return None # Return None to indicate failure
    try:
        df = pd.read_excel(DATASET_PATH) # Read Excel file instead of CSV
    except Exception as e:
        st.error(f"Error reading dataset file: {{e}}")
        return None

    documents = [Document(page_content=f"Q: {{row['prompt']}}\nA: {{row['response']}}") for _, row in df.iterrows()]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    st.success("Knowledgebase created and saved!")
    return vectorstore # Return the created vectorstore


def get_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(VECTOR_DB_PATH):
        st.warning("Knowledgebase not found. Creating from dataset...")
        vectorstore = create_vector_db()
        if vectorstore is None:
            return None # Creation failed
        return vectorstore
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def get_qa_chain(llm):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Could not load or create knowledgebase.")
        return None
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Explicitly create LLMChain and StuffDocumentsChain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context" # Ensure this matches the prompt template variable
    )

    return RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain,
        return_source_documents=False # Or True if you want to return source documents
    )

def determine_llm(question: str):
    grok_keywords = ["grok", "xai", "x-ai", "grok model", "grok chat"]
    if any(word in question.lower() for word in grok_keywords) and ChatXAI and GROK_API_KEY: # Check if ChatXAI was successfully imported and API key is available
        return "grok"
    return "gemini"

def save_chat_entry_to_csv(timestamp, name, email, sender, message):
    """Saves a single chat entry to a CSV file."""
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)

    # Check if the file exists to write header only once
    file_exists = os.path.exists(CHAT_HISTORY_FILE)

    with open(CHAT_HISTORY_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Name', 'Email', 'Sender', 'Message'])
        writer.writerow([timestamp, name, email, sender, message])
    st.write("Chat entry saved to CSV.")


def run_app():
    st.set_page_config(page_title="Gerryson Mehta Multi-LLM Chatbot", page_icon="🤖")
    st.title("Gerryson Mehta's Chatbot 🤖")
    st.write("Powered by Google Gemini and Grok AI")

    # Remove the button to create knowledgebase
    # if st.button("Create Knowledgebase from CSV"):
    #     with st.spinner("Building knowledgebase..."):
    #         create_vector_db()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_info_collected" not in st.session_state:
        st.session_state.user_info_collected = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "session_timestamp" not in st.session_state:
        st.session_state.session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # Collect user info if not already collected
    if not st.session_state.user_info_collected:
        with st.form("user_info_form"):
            st.write("Please provide your name and email to start the chat.")
            name = st.text_input("Name")
            email = st.text_input("Email")
            submit_button = st.form_submit_button("Start Chat")

            if submit_button:
                st.session_state.user_name = name
                st.session_state.user_email = email
                st.session_state.user_info_collected = True
                st.rerun() # Rerun to hide the input fields

    if st.session_state.user_info_collected:
        st.write(f"Welcome, {{st.session_state.user_name}}!")
        question = st.chat_input("Ask your career or customer service question here:")

        if question:
            llm_choice = determine_llm(question)

            llm = None # Initialize llm to None
            if llm_choice == "gemini":
                if not GOOGLE_API_KEY:
                     st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
                else:
                    try:
                        llm = GoogleGenerativeAI(
                            model="gemini-1.5-flash-latest",
                            temperature=0.3,
                            google_api_key=GOOGLE_API_KEY,
                            # system_instructions=SYSTEM_INSTRUCTION # Removed as it caused a warning
                        )
                    except Exception as e:
                        st.error(f"Error initializing Gemini model: {{e}}.")
                        print(f"Error initializing Gemini model: {{e}}") # Add logging
            elif llm_choice == "grok" and ChatXAI and GROK_API_KEY:
                 try:
                    llm = ChatXAI(
                        model="grok-4",
                        temperature=0.3,
                        xai_api_key=GROK_API_KEY,
                        # system_instructions=SYSTEM_INSTRUCTION # Removed as it caused a warning
                    )
                 except Exception as e:
                     st.error(f"Error initializing Grok model: {{e}}. Falling back to Gemini.")
                     print(f"Error initializing Grok model: {{e}}. Falling back to Gemini.") # Add logging
                     llm_choice = "gemini" # Fallback to Gemini
                     if not GOOGLE_API_KEY:
                          st.error("Google API Key not found for fallback. Please set the GOOGLE_API_KEY environment variable.")
                     else:
                         try:
                             llm = GoogleGenerativeAI(
                                model="gemini-1.5-flash-latest",
                                temperature=0.3,
                                google_api_key=GOOGLE_API_KEY,
                                # system_instructions=SYSTEM_INSTRUCTION # Removed as it caused a warning
                            )
                         except Exception as gemini_e:
                             st.error(f"Error initializing fallback Gemini model: {{gemini_e}}.")
                             print(f"Error initializing fallback Gemini model: {{gemini_e}}") # Add logging
                             llm = None # Ensure llm is None if fallback also fails
            else: # Fallback to Gemini if Grok is chosen but ChatXAI is not available or API key is missing
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
                            # system_instructions=SYSTEM_INSTRUCTION # Removed as it caused a warning
                        )
                    except Exception as e:
                        st.error(f"Error initializing fallback Gemini model: {{e}}.")
                        print(f"Error initializing fallback Gemini model: {{e}}") # Add logging
                        llm = None # Ensure llm is None if fallback also fails


            if llm: # Only proceed if an LLM was successfully initialized (or fell back to Gemini)
                qa_chain = get_qa_chain(llm)
                if qa_chain:
                    with st.spinner(f"Using {{llm_choice.title()}} model to answer..."): 
                        try:
                            response = qa_chain.invoke({"query": question})
                            answer = response.get("result", "Could not get an answer from the model.")
                        except Exception as e:
                            answer = f"An error occurred while processing your request: {{e}}"
                            st.error(answer)
                            print(f"Error during QA chain invocation: {{e}}") # Add logging
                else:
                     answer = "Sorry, I could not load the knowledge base to answer your question."

                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("assistant", answer))

                # Save chat history to CSV
                save_chat_entry_to_csv(
                    st.session_state.session_timestamp,
                    st.session_state.user_name,
                    st.session_state.user_email,
                    "User",
                    question
                )
                save_chat_entry_to_csv(
                    st.session_state.session_timestamp,
                    st.session_state.user_name,
                    st.session_state.user_email,
                    "Assistant",
                    answer
                )


            else: # Should not happen with fallback if API key is set, but as a safeguard
                 st.session_state.chat_history.append(("user", question))
                 st.session_state.chat_history.append(("assistant", "Sorry, I encountered an issue with the language model and cannot answer your question at this time. Please ensure your API keys are correctly set."))

                 # Attempt to save the user query even if LLM failed
                 save_chat_entry_to_csv(
                     st.session_state.session_timestamp,
                     st.session_state.user_name,
                     st.session_state.user_email,
                     "User",
                     question
                 )


        # Display chat history
        for sender, text in st.session_state.chat_history:
            with st.chat_message(sender):
                st.markdown(text)

run_app()