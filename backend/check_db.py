import sqlite3

db_path = r"C:\Trinetra\backend\data\trinetra.db"

conn = sqlite3.connect(db_path)

cursor = conn.cursor()

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name
""")

tables = cursor.fetchall()

print("DATABASE TABLES:")
for table in tables:
    print(table[0])

conn.close()