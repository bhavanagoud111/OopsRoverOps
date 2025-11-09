#!/usr/bin/env python3
"""Initialize database tables"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connection import init_db, check_db_connection

if __name__ == "__main__":
    print("🔧 Initializing database...")
    
    if not check_db_connection():
        print("❌ Database connection failed. Please ensure MySQL is running.")
        sys.exit(1)
    
    try:
        init_db()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)

