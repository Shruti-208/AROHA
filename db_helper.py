import sqlite3
from sqlite3 import Error

DB_NAME = "feedback.db"

def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
    except Error as e:
        print(e)
    return conn

def create_table():
    """Create feedback table if it doesn't exist."""
    conn = create_connection()
    if conn is not None:
        try:
            sql_create_feedback_table = """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                correct_label TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor = conn.cursor()
            cursor.execute(sql_create_feedback_table)
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()

def insert_feedback(text, correct_label):
    """Insert a new feedback record into the feedback table."""
    conn = create_connection()
    if conn is not None:
        try:
            sql_insert = """
            INSERT INTO feedback (text, correct_label)
            VALUES (?, ?);
            """
            cursor = conn.cursor()
            cursor.execute(sql_insert, (text, correct_label))
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()

def fetch_all_feedback():
    """Fetch all feedback records from the feedback table."""
    conn = create_connection()
    feedbacks = []
    if conn is not None:
        try:
            sql_select = "SELECT text, correct_label FROM feedback;"
            cursor = conn.cursor()
            cursor.execute(sql_select)
            feedbacks = cursor.fetchall()
        except Error as e:
            print(e)
        finally:
            conn.close()
    return feedbacks

# Additional tables for sender preferences and visit counts

def create_sender_preferences_table():
    conn = create_connection()
    if conn is not None:
        try:
            sql_create_sender_preferences = """
            CREATE TABLE IF NOT EXISTS sender_preferences (
                sender_email TEXT PRIMARY KEY,
                preferred_category TEXT NOT NULL
            );
            """
            cursor = conn.cursor()
            cursor.execute(sql_create_sender_preferences)
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()

def create_sender_visit_counts_table():
    conn = create_connection()
    if conn is not None:
        try:
            sql_create_sender_visit_counts = """
            CREATE TABLE IF NOT EXISTS sender_visit_counts (
                sender_email TEXT PRIMARY KEY,
                visit_count INTEGER NOT NULL DEFAULT 0
            );
            """
            cursor = conn.cursor()
            cursor.execute(sql_create_sender_visit_counts)
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()

def upsert_sender_preference(sender_email, preferred_category):
    conn = create_connection()
    if conn is not None:
        try:
            sql_upsert = """
            INSERT INTO sender_preferences (sender_email, preferred_category)
            VALUES (?, ?)
            ON CONFLICT(sender_email) DO UPDATE SET preferred_category=excluded.preferred_category;
            """
            cursor = conn.cursor()
            cursor.execute(sql_upsert, (sender_email, preferred_category))
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()

def get_sender_preference(sender_email):
    conn = create_connection()
    preferred_category = None
    if conn is not None:
        try:
            sql_select = "SELECT preferred_category FROM sender_preferences WHERE sender_email = ?;"
            cursor = conn.cursor()
            cursor.execute(sql_select, (sender_email,))
            row = cursor.fetchone()
            if row:
                preferred_category = row[0]
        except Error as e:
            print(e)
        finally:
            conn.close()
    return preferred_category

def increment_sender_visit_count(sender_email):
    conn = create_connection()
    if conn is not None:
        try:
            sql_insert = """
            INSERT INTO sender_visit_counts (sender_email, visit_count)
            VALUES (?, 1)
            ON CONFLICT(sender_email) DO UPDATE SET visit_count = visit_count + 1;
            """
            cursor = conn.cursor()
            cursor.execute(sql_insert, (sender_email,))
            conn.commit()
        except Error as e:
            print(e)
        finally:
            conn.close()

def get_sender_visit_count(sender_email):
    conn = create_connection()
    visit_count = 0
    if conn is not None:
        try:
            sql_select = "SELECT visit_count FROM sender_visit_counts WHERE sender_email = ?;"
            cursor = conn.cursor()
            cursor.execute(sql_select, (sender_email,))
            row = cursor.fetchone()
            if row:
                visit_count = row[0]
        except Error as e:
            print(e)
        finally:
            conn.close()
    return visit_count

# Create new tables on module import
create_sender_preferences_table()
create_sender_visit_counts_table()

# Ensure feedback table is created on module import
create_table()
