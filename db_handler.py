import sqlite3
from datetime import datetime

class ChatDB:
    def __init__(self, db_path="chat_data.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT,
                username TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

    def add_chat(self, channel, username, message):
        with self.conn:
            self.conn.execute("""
            INSERT INTO chats (channel, username, message)
            VALUES (?, ?, ?)
            """, (channel, username, message))

    def get_all_messages(self):
        with self.conn:
            cursor = self.conn.execute("SELECT channel, username, message FROM chats ORDER BY id")
            return [self._normalize_text(row[2]) for row in cursor.fetchall()]

    def _normalize_text(self, text):
        return text.lower().strip()

# Singleton instance
_db = ChatDB()

# Public API
def insert_message(message, channel="slack", username="user"):
    _db.add_chat(channel, username, message)

def get_all_messages():
    return _db.get_all_messages()
