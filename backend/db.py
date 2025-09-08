import mysql.connector
from mysql.connector import Error
import time
import os

def get_connection(max_retries=3, delay=2):
    config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'mysql123',
        'database': 'interview',
        # 'auth_plugin': 'mysql_native_password',
        # 'raise_on_warnings': True
    }
    
    for attempt in range(max_retries):
        try:
            connection = mysql.connector.connect(**config)
            
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("SELECT DATABASE()")
                db_name = cursor.fetchone()[0]
                cursor.close()
                print(f"Successfully connected to database: {db_name}")
                return connection
                
        except Error as err:
            print(f"Attempt {attempt + 1}/{max_retries}: Error connecting to database: {err}")
            
            if err.errno == 1045:  
                print("Authentication failed. Verifying user exists...")
                verify_user()
            elif err.errno == 1049: 
                print("Creating database 'interview'...")
                create_database()
            elif err.errno == 2003: 
                print("Cannot connect to MySQL server. Checking service status...")
                os.system('sudo systemctl status mysql')
                
            time.sleep(delay)
            continue
            
    print("Failed to connect after maximum retries")
    return None

def verify_user():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysql123',
        )
        cursor = connection.cursor()
        cursor.execute("SELECT User FROM mysql.user WHERE User='interview_user'")
        if not cursor.fetchone():
            print("User 'interview_user' does not exist. Please create the user.")
        cursor.close()
        connection.close()
    except Error as err:
        print(f"Error verifying user: {err}")

def create_database():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='mysql123',
        )
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS interview")
        
        cursor.execute("USE interview")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(50) DEFAULT 'user',
                interview_done BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.close()
        connection.close()
        print("Database and tables created successfully")
    except Error as err:
        print(f"Error creating database: {err}")
