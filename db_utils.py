# db_utils.py
import os
import logging
from datetime import datetime
import mysql.connector
from mysql.connector import Error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MYSQL_HOST = os.environ.get("MYSQLHOST")
MYSQL_PORT = int(os.environ.get("MYSQLPORT", 3306))
MYSQL_DATABASE = os.environ.get("MYSQLDATABASE")
MYSQL_USER = os.environ.get("MYSQLUSER")
MYSQL_PASSWORD = os.environ.get("MYSQLPASSWORD")


def _env_ok():
    return all([MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD])


def get_db_connection():
    if not _env_ok():
        logger.warning("MySQL env vars missing. Set MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD.")
        return None

    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset="utf8mb4",
            use_unicode=True,
            autocommit=False,
        )
        return conn
    except Error as err:
        logger.exception("DB connection error: %s", err)
        return None


def create_chat_history_table():
    """
    Ensure chat_history table exists. Returns True on success, False otherwise.
    """
    conn = get_db_connection()
    if not conn:
        return False

    create_sql = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255),
        session_timestamp DATETIME,
        email VARCHAR(255),
        user_question TEXT,
        assistant_answer LONGTEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_mobile VARCHAR(20)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        return True
    except Error as err:
        logger.exception("Error creating chat_history table: %s", err)
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if cur:
            cur.close()
        conn.close()


def save_chat_entry_to_db(session_timestamp, name, email, user_question, assistant_answer, user_mobile):
    """
    Insert a chat entry including user_mobile at the very end (after created_at)
    """
    conn = get_db_connection()
    if not conn:
        return False

    insert_sql = """
    INSERT INTO chat_history (session_timestamp, name, email, user_question, assistant_answer, created_at, user_mobile)
    VALUES (%s, %s, %s, %s, %s, NOW(), %s)
    """
    cur = None
    try:
        if isinstance(session_timestamp, str):
            try:
                ts = datetime.strptime(session_timestamp, "%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = datetime.now()
        elif isinstance(session_timestamp, datetime):
            ts = session_timestamp
        else:
            ts = datetime.now()

        cur = conn.cursor()
        cur.execute(insert_sql, (ts, name, email, user_question, assistant_answer, user_mobile))
        conn.commit()
        return True
    except Error as err:
        logger.exception("Error inserting chat entry: %s", err)
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if cur:
            cur.close()
        conn.close()


def test_connection():
    conn = get_db_connection()
    if not conn:
        return False
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        return result is not None
    except Exception as e:
        logger.exception("DB test failed: %s", e)
        return False
    finally:
        if cur:
            cur.close()
        conn.close()
