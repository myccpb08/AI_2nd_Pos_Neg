import sqlite3
conn = sqlite3.connect('app.db')

c = conn.cursor()

c.execute('''CREATE TABLE search_history
             (created_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              id INTEGER PRIMARY KEY,
              query text)''')

conn.commit()
conn.close()