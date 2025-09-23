# db_utils.py
import os
import logging
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Railway MySQL environment variables
MYSQL_HOST = os.environ.get("MYSQLHOST")
MYSQL_PORT = int(os.environ.get("MYSQLPORT", 3306))
MYSQL_DATABASE = os.environ.get("MYSQLDATABASE")
MYSQL_USER = os.environ.get("MYSQLUSER")
MYSQL_PASSWORD = os.environ.get("MYSQLPASSWORD")


def _env_ok():
    """Check if all required MySQL environment variables are set"""
    return all([MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD])


def get_db_connection():
    """Get database connection for Railway MySQL"""
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


def create_users_table():
    """Create users table for storing user information"""
    conn = get_db_connection()
    if not conn:
        return False

    create_sql = """
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
    
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        logger.info("users table created/verified successfully")
        return True
    except Error as err:
        logger.exception("Error creating users table: %s", err)
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if cur:
            cur.close()
        conn.close()


def create_user_sessions_table():
    """Create user_sessions table for tracking user login sessions"""
    conn = get_db_connection()
    if not conn:
        return False

    create_sql = """
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
    
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        logger.info("user_sessions table created/verified successfully")
        return True
    except Error as err:
        logger.exception("Error creating user_sessions table: %s", err)
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if cur:
            cur.close()
        conn.close()


def create_chat_history_table():
    """
    Create chat_history table with mobile number support and user_id reference.
    Updated to include foreign key to users table.
    """
    conn = get_db_connection()
    if not conn:
        return False

    create_sql = """
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
        INDEX idx_session_timestamp (session_timestamp),
        INDEX idx_user_email (user_email)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        logger.info("chat_history table created/verified successfully")
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


def create_or_get_user(full_name, email, mobile):
    """Create new user or get existing user by email"""
    conn = get_db_connection()
    if not conn:
        logger.warning("Cannot create/get user - no database connection")
        return None
        
    cur = None
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
            
            logger.info(f"Updated existing user: {email}")
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
            
            logger.info(f"Created new user: {email}")
            return {
                'user_id': user_id,
                'full_name': full_name,
                'email': email,
                'mobile': mobile,
                'last_login': datetime.now(),
                'is_new': True
            }
            
    except Error as err:
        logger.exception("Error in create_or_get_user: %s", err)
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        if cur:
            cur.close()
        conn.close()


def create_user_session(user_id):
    """Create a new user session record"""
    conn = get_db_connection()
    if not conn:
        logger.warning("Cannot create user session - no database connection")
        return None
        
    cur = None
    try:
        cur = conn.cursor()
        session_id = str(uuid.uuid4())
        
        insert_sql = """
        INSERT INTO user_sessions (session_id, user_id)
        VALUES (%s, %s)
        """
        cur.execute(insert_sql, (session_id, user_id))
        conn.commit()
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
        
    except Error as err:
        logger.exception("Error creating user session: %s", err)
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        if cur:
            cur.close()
        conn.close()


def get_user_by_email(email):
    """Get user by email address"""
    conn = get_db_connection()
    if not conn:
        return None
        
    cur = None
    try:
        cur = conn.cursor()
        select_sql = "SELECT user_id, full_name, email, mobile, created_at, last_login FROM users WHERE email = %s"
        cur.execute(select_sql, (email,))
        user = cur.fetchone()
        
        if user:
            return {
                'user_id': user[0],
                'full_name': user[1],
                'email': user[2],
                'mobile': user[3],
                'created_at': user[4],
                'last_login': user[5]
            }
        return None
        
    except Error as err:
        logger.exception("Error getting user by email: %s", err)
        return None
    finally:
        if cur:
            cur.close()
        conn.close()


def get_user_chat_history(user_id, limit=50):
    """Get chat history for a specific user"""
    conn = get_db_connection()
    if not conn:
        return []
        
    cur = None
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
        
        return [{
            'question': row[0],
            'answer': row[1],
            'timestamp': row[2]
        } for row in rows]
        
    except Error as err:
        logger.exception("Error getting user chat history: %s", err)
        return []
    finally:
        if cur:
            cur.close()
        conn.close()


def get_user_stats(user_id):
    """Get user statistics"""
    conn = get_db_connection()
    if not conn:
        return {}
        
    cur = None
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
        
        return {
            'total_chats': total_chats,
            'first_chat': first_chat,
            'last_chat': last_chat
        }
        
    except Error as err:
        logger.exception("Error getting user stats: %s", err)
        return {}
    finally:
        if cur:
            cur.close()
        conn.close()


def save_chat_entry_to_db(session_timestamp, user_name, user_email, user_mobile, user_question, assistant_answer):
    """
    Insert chat entry with mobile number and user_id reference.
    Updated with better error handling and debugging.
    """
    logger.info(f"Starting save_chat_entry_to_db for email: {user_email}")
    
    conn = get_db_connection()
    if not conn:
        logger.warning("Cannot save chat entry - no database connection")
        return False

    cur = None
    try:
        cur = conn.cursor()
        
        # Get user_id from email with detailed logging
        logger.info(f"Looking up user_id for email: {user_email}")
        cur.execute("SELECT user_id FROM users WHERE email = %s", (user_email,))
        result = cur.fetchone()
        
        user_id = None
        if result:
            user_id = result[0]
            logger.info(f"Found user_id: {user_id}")
        else:
            logger.warning(f"No user found for email: {user_email}")
            # Check if any users exist
            cur.execute("SELECT COUNT(*) FROM users")
            user_count = cur.fetchone()[0]
            logger.info(f"Total users in database: {user_count}")
            
            # List all users for debugging
            cur.execute("SELECT email, user_id FROM users")
            all_users = cur.fetchall()
            logger.info(f"All users in DB: {all_users}")
        
        # Handle timestamp
        if isinstance(session_timestamp, str):
            try:
                ts = datetime.strptime(session_timestamp, "%Y-%m-%d %H:%M:%S")
                logger.info(f"Parsed timestamp: {ts}")
            except Exception as e:
                logger.warning(f"Failed to parse timestamp '{session_timestamp}': {e}")
                ts = datetime.now()
        else:
            ts = session_timestamp or datetime.now()

        # Log data being inserted
        logger.info(f"Inserting chat data:")
        logger.info(f"  - session_timestamp: {ts}")
        logger.info(f"  - user_id: {user_id}")
        logger.info(f"  - user_name: {user_name}")
        logger.info(f"  - user_email: {user_email}")
        logger.info(f"  - user_mobile: {user_mobile}")
        logger.info(f"  - question length: {len(user_question) if user_question else 0}")
        logger.info(f"  - answer length: {len(assistant_answer) if assistant_answer else 0}")

        # Insert chat entry
        insert_sql = """
        INSERT INTO chat_history 
        (session_timestamp, user_id, user_name, user_email, user_mobile, user_question, assistant_answer)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cur.execute(insert_sql, (ts, user_id, user_name, user_email, user_mobile, user_question, assistant_answer))
        
        # Check if the insert actually inserted a row
        if cur.rowcount > 0:
            conn.commit()
            logger.info(f"Chat entry saved successfully for user: {user_email} (ID: {user_id})")
            return True
        else:
            logger.warning("No rows were inserted")
            conn.rollback()
            return False
            
    except Error as err:
        logger.exception(f"MySQL Error inserting chat entry: {err}")
        logger.error(f"Error code: {err.errno}")
        logger.error(f"SQL state: {err.sqlstate}")
        logger.error(f"Error message: {err.msg}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    except Exception as e:
        logger.exception(f"General error inserting chat entry: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if cur:
            cur.close()
        conn.close()


def get_all_users(limit=100):
    """Get all users with basic info"""
    conn = get_db_connection()
    if not conn:
        return []
        
    cur = None
    try:
        cur = conn.cursor()
        select_sql = """
        SELECT user_id, full_name, email, mobile, created_at, last_login, is_active
        FROM users 
        ORDER BY created_at DESC 
        LIMIT %s
        """
        cur.execute(select_sql, (limit,))
        rows = cur.fetchall()
        
        return [{
            'user_id': row[0],
            'full_name': row[1],
            'email': row[2],
            'mobile': row[3],
            'created_at': row[4],
            'last_login': row[5],
            'is_active': row[6]
        } for row in rows]
        
    except Error as err:
        logger.exception("Error getting all users: %s", err)
        return []
    finally:
        if cur:
            cur.close()
        conn.close()


def update_user_profile(user_id, **kwargs):
    """Update user profile data"""
    conn = get_db_connection()
    if not conn:
        return False
        
    cur = None
    try:
        cur = conn.cursor()
        
        # Build dynamic update query
        valid_fields = ['full_name', 'mobile', 'username', 'profile_data']
        updates = []
        values = []
        
        for field, value in kwargs.items():
            if field in valid_fields and value is not None:
                updates.append(f"{field} = %s")
                values.append(value)
        
        if not updates:
            return True  # Nothing to update
        
        values.append(user_id)
        update_sql = f"""
        UPDATE users 
        SET {', '.join(updates)}, updated_at = NOW()
        WHERE user_id = %s
        """
        
        cur.execute(update_sql, values)
        conn.commit()
        
        logger.info(f"Updated profile for user: {user_id}")
        return cur.rowcount > 0
        
    except Error as err:
        logger.exception("Error updating user profile: %s", err)
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if cur:
            cur.close()
        conn.close()


def initialize_all_tables():
    """Initialize all database tables"""
    success = True
    
    # Create users table first (referenced by other tables)
    if not create_users_table():
        logger.error("Failed to create users table")
        success = False
    
    # Create user sessions table
    if not create_user_sessions_table():
        logger.error("Failed to create user_sessions table")
        success = False
    
    # Create chat history table (with foreign key to users)
    if not create_chat_history_table():
        logger.error("Failed to create chat_history table")
        success = False
    
    if success:
        logger.info("All database tables initialized successfully")
    else:
        logger.warning("Some database tables failed to initialize")
    
    return success


def test_connection():
    """Test database connection"""
    conn = get_db_connection()
    if not conn:
        return False
    
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        logger.info("Database connection test successful")
        return result is not None
    except Exception as e:
        logger.exception("DB test failed: %s", e)
        return False
    finally:
        if cur:
            cur.close()
        conn.close()


def get_database_info():
    """Get database and table information"""
    conn = get_db_connection()
    if not conn:
        return {}
        
    cur = None
    try:
        cur = conn.cursor()
        
        # Get table information
        cur.execute("SHOW TABLES")
        tables = [row[0] for row in cur.fetchall()]
        
        table_info = {}
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            table_info[table] = count
        
        return {
            'database': MYSQL_DATABASE,
            'tables': table_info,
            'total_tables': len(tables)
        }
        
    except Error as err:
        logger.exception("Error getting database info: %s", err)
        return {}
    finally:
        if cur:
            cur.close()
        conn.close()


# For backward compatibility
def create_table():
    """Deprecated: Use create_chat_history_table() instead"""
    logger.warning("create_table() is deprecated. Use create_chat_history_table() instead.")
    return create_chat_history_table()
