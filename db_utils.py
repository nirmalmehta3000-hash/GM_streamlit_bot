import mysql.connector
import streamlit as st
import os

MYSQL_HOST = os.environ.get("MYSQL_HOST")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE")
MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            database=MYSQL_DATABASE,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"DB connection error: {err}")
        return None

def create_chat_history_table():
    conn = get_db_connection()
    if not conn:
        return
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            name VARCHAR(255),
            timestamp DATETIME,
            email VARCHAR(255),
            user_question TEXT,
            assistant_answer TEXT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def save_chat_entry_to_db(timestamp, name, email, user_question, assistant_answer):
    conn = get_db_connection()
    if not conn:
        return
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chat_history (name, timestamp, email, user_question, assistant_answer)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (name, timestamp, email, user_question, assistant_answer),
    )
    conn.commit()
    cursor.close()
    conn.close()
