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
from torch.utils.data import Dataset


_OPTICAL_MAX_VALUE = 2000. # Magic number some guys at Google figured out. Don't touch.


class BigEarthNetDataset(Dataset):
    def __init__(self, data_dir="../BigEarthNet-v1.0/", filter_files=["../patches_with_cloud_and_shadow.csv", "../patches_with_seasonal_snow.csv"], filter_data=True, mode='rgb', label_count_cache='./label_counts.pkl', val_prop=0.25, test_prop=0.2, split_file=None, split_save_path='splits.pkl', seed=42):
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


    def get_train_val_test_indices(self, premade_split_file=None, split_save_path=None, rebuild=False):
        if hasattr(self, 'train_indices') and hasattr(self, 'val_indices') and hasattr(self, 'test_indices') and not rebuild:
            return self.train_indices, self.val_indices, self.test_indices
        train_indices, val_indices, test_indices = [], [], []
        if premade_split_file and not rebuild:
            print("Reloading train-val-test split cache from", premade_split_file)
            with open(premade_split_file, 'rb') as f:
                indices = pickle.load(f)
                self.train_keys, self.validation_keys, self.test_keys = indices['train_keys'], indices['val_keys'], indices['test_keys']
                return indices['train'], indices['val'], indices['test']
        if split_save_path is None: warnings.warn("Building index list, but not saving!")
        print("Building new train-val-test split and saving to", split_save_path)
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


    def __getitem__(self, idx):
        # load image
        band_stack = []
        for bands in self.bands:
            band_path = os.path.join(self.data_dir, self.patches[idx], "{}_{}.tif".format(self.patches[idx], bands))
            assert os.path.isfile(band_path)
            band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = raster_band.ReadAsArray()
            band_stack.append(band_data)
        if self.mode == 'rgb':
            img = np.stack(band_stack) / _OPTICAL_MAX_VALUE # (C, W, H)
            img = np.clip(img, 0, 1)
        else:
            raise NotImplementedError()
        with open(os.path.join(self.data_dir, self.patches[idx], "{}_labels_metadata.json".format(self.patches[idx])), 'r') as f:
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
        return img, labels


    def __len__(self):
        return len(patches)


    def translate_label_vector(self, labels):
        names = []
        one_indices = np.where(labels != 0)[0] 
        for i in one_indices:
            names.append(self.idx_to_label[i])
        return names
