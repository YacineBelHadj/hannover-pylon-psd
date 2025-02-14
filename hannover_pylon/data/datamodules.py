from pathlib import Path
import sqlite3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

# =============================================================================
# Custom Dataset
# =============================================================================
class PSDDataset(Dataset):
    def __init__(self, db_path, table_name, columns, transform_func, query_key, return_dict=True, cached=False):
        """
        Args:
            db_path (str or Path): Path to the database file.
            table_name (str): Name of the table to query.
            columns (list[str]): List of column names to retrieve.
            transform_func (list[nn.Module]): A list of transformation modules (one per column).
            query_key (str): SQL query to select row identifiers (here, ids).
            return_dict (bool): Whether to return a dictionary or tuple per sample.
            cached (bool): If True, load and process all rows into memory.
        """
        self.db_path = str(db_path)  # Ensure it's a string for sqlite3.
        self.table_name = table_name
        self.columns = columns
        self.transform_func = transform_func
        self.return_dict = return_dict
        self.cached = cached

        # Query the database to get the list of row ids based on query_key.
        self.indices = self._get_ids(query_key)
        if self.cached:
            # Preload and process all data if caching is enabled.
            self.data = [self._process_item(self._load_item(item_id)) for item_id in self.indices]
        else:
            self.data = None

    def _get_ids(self, query_key):
        """Execute query_key to get a list of row identifiers."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query_key)
        rows = cursor.fetchall()
        conn.close()
        # Assume that each row contains one value: the id.
        return [row[0] for row in rows]

    def _load_item(self, item_id):
        """Query the database for a single row by its id."""
        conn = sqlite3.connect(self.db_path)
        # Use a Row factory so we can access columns by name.
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Build a query that fetches only the desired columns.
        query = f"SELECT {', '.join(self.columns)} FROM {self.table_name} WHERE id = ?"
        cursor.execute(query, (item_id,))
        row = cursor.fetchone()
        conn.close()
        if row is None:
            raise ValueError(f"No data found for id {item_id}")
        # Convert the sqlite3.Row to a regular dictionary.
        return {col: row[col] for col in self.columns}

    def _process_item(self, raw_item):
        """Apply the per-column transforms to the raw item."""
        processed_item = {}
        for col, tf in zip(self.columns, self.transform_func):
            raw_val = raw_item[col]
            processed_val = tf(raw_val)
            processed_item[col] = processed_val
        return processed_item

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.cached:
            # Return the cached, processed item.
            processed_item = self.data[index]
        else:
            # Load the raw item and process it on the fly.
            item_id = self.indices[index]
            raw_item = self._load_item(item_id)
            processed_item = self._process_item(raw_item)
        if self.return_dict:
            return processed_item
        else:
            # Return a tuple in the same order as self.columns.
            return tuple(processed_item[col] for col in self.columns)


# =============================================================================
# DataModule
# =============================================================================
class PSDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        db_path,
        table_name,
        columns,
        transform_func,
        query_key,
        batch_size=32,
        return_dict=True,
        cached=False,
        num_workers=0,
    ):
        """
        Args:
            db_path (str or Path): Path to the database file.
            table_name (str): Table to query.
            columns (list[str]): List of column names to load.
            transform_func (list[nn.Module]): List of transforms, one per column.
            query_key (str): SQL query for selecting the row ids.
            batch_size (int): Batch size for training.
            return_dict (bool): Whether each sample is returned as a dict.
            cached (bool): Whether to cache all data in memory.
            num_workers (int): Number of DataLoader workers.
        """
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.columns = columns
        self.transform_func = transform_func
        self.query_key = query_key
        self.batch_size = batch_size
        self.return_dict = return_dict
        self.cached = cached
        self.num_workers = num_workers
        self.is_setup = False

    def setup(self, stage=None):
        if not self.is_setup:
            self._setup()
            self.is_setup = True
        else :
            print("DataModule already setup")
    def _setup(self, stage=None):
        # For this example, we only implement the training dataset.
        self.dataset = PSDDataset(
            db_path=self.db_path,
            table_name=self.table_name,
            columns=self.columns,
            transform_func=self.transform_func,
            query_key=self.query_key,
            return_dict=self.return_dict,
            cached=self.cached,
        )
        ds_len = len(self.dataset)
        tr_len = int(0.8 * ds_len)
        val_len = ds_len - tr_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, [tr_len, val_len])
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    def all_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if True:
    if __name__=='__main__':
        from config import settings
        from pathlib import Path
        from torch import nn

        db_path = Path(settings.path.processed, 'Welch(n_fft=16392, fs=1651, max_freq=825.5).db')

        query_key = f'''
            SELECT id FROM data
            WHERE sensor = "accel"
            AND date BETWEEN "{settings.state.healthy_train.start}" AND "{settings.state.healthy_train.end}"
            AND corrupted = 0
        '''

        columns = ['psd', 'date', 'sensor', 'level', 'direction']
        
        transform_func = [ nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()]
        
        data_module = PSDDataModule(
            db_path=db_path,
            query_key=query_key,
            columns=columns,
            transform_func=transform_func,
            batch_size=32,
            num_workers=4,
            return_dict=True
        )
        data_module.setup()
        for batch in data_module.train_dataloader():
            print(batch)
            break
        
if __name__ == '__main__':
    from config import settings
    from pathlib import Path
    from torch import nn

    db_path = Path(settings.path.processed, 'Welch(n_fft=16392, fs=1651, max_freq=825.5).db')

    query_key = f'''
        SELECT DISTINCT date
        FROM data
        WHERE sensor = "accel"
        AND date BETWEEN "{settings.state.healthy_train.start}" AND "{settings.state.healthy_train.end}"
        AND corrupted = 0
        ORDER BY date
    '''

    columns = ['psd','date']
    transform_func = [nn.Identity(), nn.Identity(), nn.Identity()]

    data_module = PSDDataModule(
        db_path=db_path,
        query_key=query_key,
        columns=columns,
        transform_func=transform_func,
        batch_size=32,
        num_workers=4,
        return_dict=True,
        
    )

    data_module.setup()
    
    for batch in data_module.train_dataloader():
        print("Batch of PSDs grouped by date:")
        print(batch['psd'][0].shape)
        
        break
