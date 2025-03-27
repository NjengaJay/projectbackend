import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def reset_database():
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='Kurtzman',
        host='localhost',
        port='5432'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Drop database if exists
    try:
        cur.execute('DROP DATABASE IF EXISTS trip_planner')
        print("Dropped existing database")
    except Exception as e:
        print(f"Error dropping database: {e}")
    
    # Create new database
    try:
        cur.execute('CREATE DATABASE trip_planner')
        print("Created new database")
    except Exception as e:
        print(f"Error creating database: {e}")
    
    cur.close()
    conn.close()

if __name__ == '__main__':
    reset_database()
