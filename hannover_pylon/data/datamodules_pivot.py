from torch.utils.data import Dataset, DataLoader, random_split, default_collate
import sqlite3
import pytorch_lightning as pl
import torch
import numpy as np
def custom_collate_fn(batch):
    """
    Custom collate function to properly batch dictionary-like outputs
    from the PSDDataset.
    """
    # Initialize a dictionary to hold batched data
    batched = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():

            batched[key].append(value)

    return batched


class PSDDataset(Dataset):
    def __init__(self, db_path, query_key, columns, transform_func=None, return_dict=False):
        self.db_path = db_path
        self.query_key = query_key
        self.columns = columns
        self.transform_func = transform_func or [lambda x: x] * len(columns)
        self.return_dict = return_dict

        # Fetch all unique dates based on the query_key
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            self.dates = c.execute(self.query_key).fetchall()
            self.dates = [date[0] for date in self.dates]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        # Get all PSDs for a specific date
        date = self.dates[idx]
        query = f'''
            SELECT {", ".join(self.columns)}
            FROM data
            WHERE date = ?
            AND sensor = "accel"
            AND corrupted = 0
        '''

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            rows = c.execute(query, (date,)).fetchall()

        # Transform each column
        transformed = [
            [func(row[i]) for row in rows]
            for i, func in enumerate(self.transform_func)
        ]
        # stack the first dimension
        transformed[0] = torch.stack(transformed[0])

        if self.return_dict:
            return {col: transformed[i] for i, col in enumerate(self.columns)}
        else:
            return transformed


class PSDDataModule(pl.LightningDataModule):
    def __init__(self, db_path, query_key, columns, transform_func=None, batch_size=32, num_workers=4, return_dict=False):
        super().__init__()
        self.db_path = db_path
        self.query_key = query_key
        self.columns = columns
        self.transform_func = transform_func or [lambda x: x] * len(columns)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_dict = return_dict
        self.dataset = None

    def setup(self, stage=None):
        self.dataset = PSDDataset(
            db_path=self.db_path,
            query_key=self.query_key,
            columns=self.columns,
            transform_func=self.transform_func,
            return_dict=self.return_dict
        )

        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn
        )
    def all_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn
        )

if __name__ == '__main__':
    # Configuration
    from config import settings
    from pathlib import Path
    from torch import nn
    from hannover_pylon.modelling.transformation import FromBuffer

    db_path = Path(settings.path.processed, 'Welch(n_fft=16392, fs=1651, max_freq=825.5).db')
    query_key = f'''
        SELECT DISTINCT date
        FROM data
        WHERE sensor = "accel"
        AND corrupted = 0
        ORDER BY date
    '''
    columns = ['psd', 'level', 'direction']
    transform_func = [FromBuffer(), nn.Identity(),nn.Identity()]

    # Initialize DataModule
    data_module = PSDDataModule(
        db_path=db_path,
        query_key=query_key,
        columns=columns,
        transform_func=transform_func,
        batch_size=32,
        num_workers=0,
        return_dict=True
    )

    data_module.setup()

    for batch in data_module.train_dataloader():
        print("Batch of PSDs grouped by date:")
        print(batch['psd'][0])
        break
