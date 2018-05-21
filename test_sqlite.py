import sqlite3 as sq

db = sq.connect('mydb.dat')
cursor = db.cursor()
cursor.execute('''
    CREATE TABLE manip(
        id INTEGER PRIMARY KEY,
        name TEXT,
    )
''')
db.commit()
db.close()
