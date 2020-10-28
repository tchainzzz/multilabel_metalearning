import os
import random

import csv
import gdal
import rasterio
import pickle
import json

from tqdm import tqdm
import warnings

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


_OPTICAL_MAX_VALUE = 2000. # Magic number some guys at Google figured out. Don't touch.


class BigEarthNetDataset(Dataset):
    def __init__(self, data_dir="../BigEarthNet-v1.0/", filter_files=["../patches_with_cloud_and_shadow.csv", "../patches_with_seasonal_snow.csv"], filter_data=True, mode='rgb', label_count_cache='./label_counts.pkl', val_prop=0.25, test_prop=0.2, split_file=None, split_save_path='splits.pkl', seed=42, meta=False):
        super(BigEarthNetDataset, self).__init__()
        random.seed(42)
        mode = mode.lower()
        if mode not in ['rgb', 'all']:
            raise Exception("Dataset mode must be 'rgb' (using only RGB channels) or 'all' (use all spectral bands.")
        self.mode = mode
        if self.mode == 'rgb':
            self.bands = ['B04', 'B03', 'B02']
        else:
            self.bands = ['B01', 'B02', 'B03', 'B04', 'B05',
                                  'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        self.data_dir = data_dir
        self.patches = os.listdir(data_dir)
        self.patches.sort()

        elimination_patch_list = []
        if filter_data:
            for file_path in filter_files:
                if not os.path.exists(file_path):
                    print('ERROR: file located at', file_path, 'does not exist')
                    exit()
                with open(file_path, 'r') as f:
                    csv_reader = csv.reader(f, delimiter=',')
                    for row in csv_reader:
                        elimination_patch_list.append(row[0])
            elimination_patch_list = set(elimination_patch_list)
            self.patches = [patch for patch in self.patches if patch not in elimination_patch_list]

        self.counts = Counter()
        if not os.path.isfile(label_count_cache):
            print("Label count cache at {} not found; rebuilding cache.".format(label_count_cache))
            for patch in tqdm(self.patches):
                with open(os.path.join(data_dir, patch, "{}_labels_metadata.json".format(patch)), 'r') as f:
                    metadata = json.load(f)
                    self.counts.update(metadata['labels'])
            with open(label_count_cache, 'wb') as cache_file:
                pickle.dump(self.counts, cache_file)
        else:
            with open(label_count_cache, 'rb') as cache_file:
                self.counts = pickle.load(cache_file)
        self.idx_to_label = dict(enumerate(sorted(self.counts.keys())))
        #self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}
        self.test_keys = set(sorted(random.choices(list(self.counts.keys()), k=int(test_prop * len(self.counts)))))
        remaining_keys = [k for k in self.counts.keys() if k not in self.test_keys]
        self.validation_keys = set(sorted(random.choices(remaining_keys, k=int(val_prop * len(remaining_keys)))))
        self.train_keys = {k for k in remaining_keys if k not in self.validation_keys}
        self.train_indices, self.val_indices, self.test_indices = self.get_train_val_test_indices(premade_split_file=split_file, split_save_path=split_save_path)
        self.meta = meta # are we using this as a part of meta-training/val/test?

    def get_train_val_test_indices(self, premade_split_file=None, split_save_path=None, rebuild=False):
        if hasattr(self, 'train_indices') and hasattr(self, 'val_indices') and hasattr(self, 'test_indices') and not rebuild:
            return self.train_indices, self.val_indices, self.test_indices
        train_indices, val_indices, test_indices = [], [], []
        if premade_split_file and not rebuild:
            print("Reloading train-val-test split cache from", premade_split_file)
            try:
                with open(premade_split_file, 'rb') as f:
                    indices = pickle.load(f)
                    self.train_keys, self.validation_keys, self.test_keys = indices['train_keys'], indices['val_keys'], indices['test_keys']
                    return indices['train'], indices['val'], indices['test']
            except OSError:
                message = ("Creating new split instead with file name {}".format(split_save_path), "Attempting to create split with this file name.")[split_save_path is None]
                print("File", premade_split_file, "not found. {}".format(message))
                if split_save_path is None:
                    split_save_path = premade_split_file
        if split_save_path is None: 
            warnings.warn("Building index list, but not saving!")
        print("Building new train-val-test split and saving to", split_save_path)
        #self.labels = []
        for i, patch in enumerate(tqdm(self.patches)):
            with open(os.path.join(self.data_dir, patch, "{}_labels_metadata.json".format(patch)), 'r') as f:
                metadata = json.load(f)
                labels = set(metadata['labels'])
                train_cardinality = len(labels & self.train_keys)
                val_cardinality = len(labels & self.validation_keys)
                test_cardinality = len(labels & self.test_keys)
                max_cardinality = max(train_cardinality, val_cardinality, test_cardinality)
                if max_cardinality == train_cardinality:
                    train_indices.append(i)
                elif max_cardinality == val_cardinality:
                    val_indices.append(i)
                else:
                    test_indices.append(i)
                #self.labels.append(labels)
        if split_save_path:
            index_dict = {'train_keys': self.train_keys,
                    'val_keys': self.validation_keys,
                    'test_keys': self.test_keys,
                    'train': train_indices,
                    'val': val_indices,
                    'test': test_indices}
            with open(split_save_path, 'wb') as f:
                pickle.dump(index_dict, f)
        return train_indices, val_indices, test_indices


    def peek_label(self, idx):
        patch = self.patches[idx]
        with open(os.path.join(self.data_dir, patch, "{}_labels_metadata.json".format(patch)), 'r') as f:
            metadata = json.load(f)
            raw_labels = set(metadata['labels'])
            return raw_labels

    def __getitem__(self, idx):
        # load image
        band_stack = []
        patch = self.patches[idx]
        for bands in self.bands:
            band_path = os.path.join(self.data_dir, patch, "{}_{}.tif".format(patch, bands))
            assert os.path.isfile(band_path)
            band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = raster_band.ReadAsArray()
            band_stack.append(band_data)
        if self.mode == 'rgb':
            img = np.stack(band_stack) / _OPTICAL_MAX_VALUE # (C, W, H)
            img = np.clip(img, 0, 1)
            img = torch.Tensor(img)
        else:
            raise NotImplementedError()

        with open(os.path.join(self.data_dir, patch, "{}_labels_metadata.json".format(patch)), 'r') as f:
            metadata = json.load(f)
            raw_labels = set(metadata['labels'])

        # load labels
        if idx in self.train_indices:
            names = raw_labels & self.train_keys
        elif idx in self.val_indices:
            names = raw_labels & self.validation_keys
        elif idx in self.test_indices:
            names = raw_labels & self.test_keys
        else:
            raise IndexError("Index not found in provided indices. Try rebuilding indices from scratch by passing rebuild=True into get_train_val_test_indices().")

        # k-hot vector of classes -> sample batches by taking 
        labels = np.array([1 if cover_type in names else 0 for cover_type in self.counts.keys()])
        if not self.meta:
            labels = torch.LongTensor(labels)
        return img, labels


    def __len__(self):
        return len(patches)


    def translate_label_vector(self, labels):
        names = []
        one_indices = np.where(labels != 0)[0] 
        for i in one_indices:
            names.append(self.idx_to_label[i])
        return names


class MetaBigEarthNetTaskDataset(IterableDataset):
    def __init__(self, split='train', support_size=4, label_subset_size=5, data_dir="../BigEarthNet-v1.0/", filter_files=["../patches_with_cloud_and_shadow.csv", "../patches_with_seasonal_snow.csv"], filter_data=True, mode='rgb', label_count_cache='./label_counts.pkl', val_prop=0.25, test_prop=0.2, split_file=None, split_save_path='splits.pkl', seed=42):
        super(MetaBigEarthNetTaskDataset, self).__init__()
        random.seed(seed)
        self.support_size = support_size
        self.label_subset_size = label_subset_size
        self.split = split
        self.dataset = BigEarthNetDataset(data_dir=data_dir, filter_files=filter_files, filter_data=filter_data, mode=mode, label_count_cache=label_count_cache, val_prop=val_prop, test_prop=test_prop, split_file=split_file, split_save_path=split_save_path, seed=seed, meta=True)
        if split not in ['train', 'val', 'test']:
            raise Exception("Invalid split; must be one of 'train', 'val', or 'test'.")
        if support_size < 2:
            raise Exception("Support set size must be at least 2.")
        if label_subset_size < 1:
            raise Exception("Subset size must be strictly positive.")

        self.split = split
        if split == 'train':
            self.indices = self.dataset.train_indices
        elif split == 'val':
            self.indices = self.dataset.val_indices
        else:
            self.indices = self.dataset.test_indices


    def __iter__(self):
        n_classes = len(self.dataset.idx_to_label)
        while True:
            seen = np.zeros((n_classes,), dtype=int)
            support = [] # target shape: (support, w, h) -> then we can collate
            labels = [] # target shape: (support, label_subset_size)
            while len(support) < self.support_size:
                idx = random.choice(self.indices)
                img, label = self.dataset[idx] # shape: (w, h), (n_classes)
                if np.count_nonzero(label | seen) > self.label_subset_size:
                    continue
                seen = label | seen
                support.append(img)
                labels.append(torch.LongTensor(label))
            support = torch.stack(support, dim=0)
            labels = torch.stack(labels, dim=0) # shape is temporariliy (support, n_classes)
            selected_class_mask = torch.abs(labels).sum(dim=0) > 0
            cardinality = np.count_nonzero(seen)
            artificial_classes = np.random.choice(np.where(~selected_class_mask)[0], size=self.label_subset_size - cardinality, replace=False)
            selected_class_mask[artificial_classes] = True
            labels = labels[:, selected_class_mask]
            yield support, labels

def get_dataloaders(train_batch_size=8, val_batch_size=8, test_batch_size=8, support_size=4, label_subset_size=5, data_dir="../BigEarthNet-v1.0/", filter_files=["../patches_with_cloud_and_shadow.csv", "../patches_with_seasonal_snow.csv"], filter_data=True, mode='rgb', label_count_cache='./label_counts.pkl', val_prop=0.25, test_prop=0.2, split_file=None, split_save_path='splits.pkl', seed=42):
    train = MetaBigEarthNetTaskDataset(split='train', support_size=support_size, label_subset_size=label_subset_size, data_dir=data_dir, filter_files=filter_files, filter_data=filter_data, mode=mode, label_count_cache=label_count_cache, val_prop=val_prop, test_prop=test_prop, split_file=split_file, split_save_path=split_save_path, seed=seed)
    val = MetaBigEarthNetTaskDataset(split='val', support_size=support_size, label_subset_size=label_subset_size, data_dir=data_dir, filter_files=filter_files, filter_data=filter_data, mode=mode, label_count_cache=label_count_cache, val_prop=val_prop, test_prop=test_prop, split_file=split_file, split_save_path=split_save_path, seed=seed)
    test = MetaBigEarthNetTaskDataset(split='test', support_size=support_size, label_subset_size=label_subset_size, data_dir=data_dir, filter_files=filter_files, filter_data=filter_data, mode=mode, label_count_cache=label_count_cache, val_prop=val_prop, test_prop=test_prop, split_file=split_file, split_save_path=split_save_path, seed=seed)
    train_dataloader = DataLoader(train, batch_size=train_batch_size)
    val_dataloader = DataLoader(val, batch_size=val_batch_size)
    test_dataloader = DataLoader(test, batch_size=test_batch_size)
    return train_dataloader, val_dataloader, test_dataloader
