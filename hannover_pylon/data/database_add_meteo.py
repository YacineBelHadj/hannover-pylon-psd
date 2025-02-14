import sqlite3
from pathlib import Path
from tqdm import tqdm
from config import settings
from hannover_pylon.data.pipeline import build_pipeline_meteo
import math

def main():
    dp_meta_meteo = build_pipeline_meteo()
    # Setup SQLite database
    db_name = "meteo_data.db"
    path_db = Path(settings.path.processed) / 'Welch(n_fft=16392, fs=1651, max_freq=825.5).db'
    assert path_db.exists(), f"Database {path_db} does not exist. Please run the database script first."
    with sqlite3.connect(path_db) as conn:
        cursor = conn.cursor()
        # delete table if it exists
        cursor.execute('DROP TABLE IF EXISTS meteo_data')

        # Create table with appropriate schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meteo_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                value REAL NOT NULL
            )
        ''')

        # Create indexes to optimize queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON meteo_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_channel_name ON meteo_data(channel_name)')

        # Prepare the SQL insert statement
        insert_query = '''
            INSERT INTO meteo_data (date, channel_name, value)
            VALUES (?, ?, ?)
        '''

        # Begin a transaction for better performance
        conn.execute('BEGIN TRANSACTION')
        for data in tqdm(dp_meta_meteo, desc="Inserting meteorological data into DB"):
            data_tuples = data['data_tuples']  # List of tuples (date, value, channelName)

            # Ensure data types are correct and match the order expected by insert_query
            processed_tuples = []
            for d, v, cn in data_tuples:
                # Check if value is None or NaN
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    # Optionally, log or print the skipped entry
                    print(f"Skipping entry with missing value: date={d}, channel={cn}")
                    continue  # Skip this entry
                processed_tuples.append((d.isoformat(), str(cn), float(v)))

            if processed_tuples:
                # Execute insert statements in batches
                cursor.executemany(insert_query, processed_tuples)
                conn.commit()  # Commit after each file (you can adjust this as needed)
        conn.execute('END TRANSACTION')

if __name__ == "__main__":
    main()