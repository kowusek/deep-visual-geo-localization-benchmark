import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as T
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from matplotlib.pyplot import imshow

base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

identity_transform = T.Lambda(lambda x: x)

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

#Dataset used to access Query data
class QueryDataset(data.Dataset):
    def __init__(self, args, datasets_folder="datasets_vg/datasets", dataset_name="VIGOR", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        queries_folder = join(self.dataset_folder, "queries")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"),  recursive=True))         
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)

    def __getitem__(self, index):
        img = path_to_pil_img(self.queries_paths[index])
        img = base_transform(img)
        return img

    def __len__(self):
        return len(self.queries_paths)

    def get_utms(self):
        return self.queries_utms

#Dataset used to access Database data
class DatabaseDataset(data.Dataset):
    def __init__(self, args, datasets_folder="datasets_vg/datasets", dataset_name="VIGOR", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        database_folder = join(self.dataset_folder, "database")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"),  recursive=True))         
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)

    def __getitem__(self, index):
        img = path_to_pil_img(self.database_paths[index])
        img = base_transform(img)
        return img

    def __len__(self):
        return len(self.database_paths)

    def get_utms(self):
        return self.database_utms

class TripletsDataset(data.Dataset):
    def __init__(self, args, datasets_folder="datasets_vg/datasets", dataset_name="VIGOR", split="train", negs_num_per_query=10):
        super().__init__()
        self.mining = args.mining
        self.query_resize = args.query_resize
        self.database_resize = args.database_resize
        self.neg_samples_num = args.neg_samples_num  # Number of negatives to randomly sample
        self.negs_num_per_query = negs_num_per_query  # Number of negatives per query in each batch
        self.q_ds = QueryDataset(args, datasets_folder=datasets_folder, dataset_name=dataset_name, split=split)
        self.db_ds = DatabaseDataset(args, datasets_folder=datasets_folder, dataset_name=dataset_name, split=split)

        self.database_transform = T.Compose([
            T.Resize(self.database_resize, antialias=True) if self.database_resize is not None else identity_transform,
            base_transform
        ])
        
        self.query_transform = T.Compose([
            T.Resize(self.query_resize, antialias=True) if self.query_resize is not None else identity_transform,
            base_transform
        ])

        nn = NearestNeighbors(n_jobs=-1)
        nn.fit(self.db_ds.get_utms())

        self.soft_positives_per_query = nn.radius_neighbors(self.q_ds.get_utms(),
                                                             radius=args.val_positive_dist_threshold,
                                                             return_distance=False)

        self.hard_positives_per_query = nn.radius_neighbors(self.q_ds.get_utms(),
                                             radius=args.train_positives_dist_threshold,
                                             return_distance=False)

        queries_with_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) != 0)[0]
        if len(queries_with_hard_positive) != len(self.q_ds):
            logging.info(f"There are {len(self.q_ds) - len(queries_with_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")

        # Remove queries without positives
        self.hard_positives_per_query = self.hard_positives_per_query[queries_with_hard_positive]
        self.soft_positives_per_query = self.soft_positives_per_query[queries_with_hard_positive]

        # Remove queries without positives from dataset
        self.q_ds = Subset(self.q_ds, queries_with_hard_positive)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass