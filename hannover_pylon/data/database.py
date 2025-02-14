import sqlite3
import pickle 
from pathlib import Path
from tqdm import tqdm
from config import settings
from hannover_pylon.data.pipeline import build_pipeline

def main():
    dp_acc, dp_meta_met, dp_meta_struct, dp_corrupted , welch= build_pipeline()
    # Setup SQLite database
    path_db = Path(settings.path.processed) / f"{str(welch)}1.db"
    path_db.parent.mkdir(parents=True, exist_ok=True)

    # Connect to SQLite database using a context manager
    with sqlite3.connect(path_db) as conn:
        cursor = conn.cursor()

        # Create table with appropriate schema
        # delte the table if it already exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                psd BLOB NOT NULL,
                date TIMESTAMP NOT NULL,
                sensor TEXT NOT NULL,
                level INT NOT NULL,
                direction TEXT NULL,
                corrupted BOOLEAN NOT NULL
            )
        ''')

        # Create indexes to optimize queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor ON data(sensor)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON data(date)')

        # Prepare the SQL insert statement
        insert_query = '''
            INSERT INTO data (psd, date, sensor,level, direction, corrupted)
            VALUES (?, ?, ?, ?, ?, ?)
        '''


        # Begin a transaction for better performance
        conn.execute('BEGIN TRANSACTION')
        for batch in tqdm(dp_acc, desc="Inserting batches into DB"):

            date_batch = batch['data']['dates']
            psd_batch = batch['data']['log_psd']
            level_batch = batch['data']['levels']
            direction_batch = batch['data']['directions']
            corrupted_batch = batch['data']['corrupted_flags']
            sensor_batch = batch['data']['sensors']

            # Serialize the PSDs using pickle
            psd_serialized = [psd.cpu().numpy().astype('float32').tobytes() for psd in psd_batch]
            cursor.executemany(insert_query, list(zip(psd_serialized, date_batch, sensor_batch,level_batch, direction_batch, corrupted_batch)))
            conn.commit()
        conn.execute('END TRANSACTION')
        

    # Optional: Verify the insertion by querying the database
    # Uncomment the following lines if you want to perform a simple verification
    """


    with sqlite3.connect(path_db) as verify_conn:
        verify_cursor = verify_conn.cursor()
        verify_cursor.execute('SELECT COUNT(*) FROM data')
        count = verify_cursor.fetchone()[0]
        print(f"Total records in the database: {count}")
    """

if __name__ == "__main__":
    main()
