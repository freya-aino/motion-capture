import tqdm
import random 
import numpy as np
import torch as T
import torch.utils.data as Tdata
import pytorch_lightning as pl

from copy import deepcopy


class BboxDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        dataset: Tdata.Dataset,
        image_augmentation,
        image_shape: tuple[int, int], # width, height
        batch_size: int,
        train_val_test_split: tuple[float, float, float],
        num_train_workers: int,
        num_val_workers: int,
        pre_load_dataset: bool):
        
        super().__init__()
        self.dataset = dataset
        self.image_augmentation = image_augmentation
        
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.num_train_workers = num_train_workers
        self.num_val_workers = num_val_workers
        self.pre_load_dataset = pre_load_dataset
        
    def setup(self):
        # Setup your dataset(s) for training, validation, test
        
        splits = Tdata.random_split(
            dataset = self.dataset, 
            lengths = self.train_val_test_split,
        )
        
        print(f"Train: {len(splits[0])}, Val: {len(splits[1])}, Test: {len(splits[2])}")
        
        if self.pre_load_dataset:
            self.train_dataset = [d for d in splits[0]]
            self.val_dataset = [d for d in splits[1]]
            self.test_dataset = [d for d in splits[2]]
        else:
            self.train_dataset = splits[0]
            self.val_dataset = splits[1]
            self.test_dataset = splits[2]
        
    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        x = T.stack([self.image_augmentation(b[0]) / 255 for b in batch]).to(dtype=T.float32)
        y = T.stack([(b[1] / T.tensor([self.image_shape] * 2).flatten(-2)) for b in batch]).to(dtype=T.float32)
        v = T.stack([b[2] for b in batch]).to(dtype=T.float32)
        return x, y, v
    
    def train_dataloader(self):
        return Tdata.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_train_workers,
            persistent_workers=True if self.num_train_workers > 0 else False,
            # generator = self.generator,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return Tdata.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            # shuffle=False,
            num_workers=self.num_val_workers,
            persistent_workers=True if self.num_val_workers > 0 else False,
            # sampler=sampler,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return Tdata.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=False,
        )
