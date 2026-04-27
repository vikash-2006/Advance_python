import mysql.connector

conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password=""        # blank password
)

if conn.is_connected():
    print("✅ Connected to MySQL successfully!")

cursor = conn.cursor()
cursor.execute("SHOW DATABASES;")
print("\n📂 Your Databases:")
for db in cursor.fetchall():
    print(" -", db[0])

cursor.close()
conn.close()

import mysql.connector

import pandas as pd

conn = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password=""
)

cursor = conn.cursor()

# Select a database — change 'world' to any you want
cursor.execute("USE world;")

# Show tables
cursor.execute("SHOW TABLES;")
print("📋 Tables:")
for table in cursor.fetchall():
    print(" -", table[0])

# Query a table using pandas
df = pd.read_sql("SELECT * FROM city LIMIT 10;", conn)
print("\n📊 Data Preview:")
print(df)

cursor.close()
conn.close()