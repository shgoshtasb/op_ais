import os
import sys
import six
import string
import argparse

import lmdb
import pickle
import msgpack
import tqdm
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


length = {
    "train": 162770,
    "val": 19867,
    "test": 19962,
}

class LMDBDataset(data.Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.db_path = os.path.join(self.root, 'celeba', f"{split}.db")
        self.env = None
        self.transform = transform
        self.length = length[split]
        
    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

    def __getitem__(self, index):
        img, target = None, None
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        #img = Image.fromarray(imgbuf.astype('uint8'), 'RGB')
        img = Image.fromarray(imgbuf, 'RGB')
        if self.transform is not None:
            img = self.transform(img)

        # load label
        target = unpacked[1]

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'



def celeba2lmdb(path, outpath, write_frequency=5000):
    dataset_dir = os.path.expanduser(path)
    print("Loading dataset from %s" % dataset_dir)
    
    transform = transforms.Compose([
                            transforms.Lambda(lambda x: np.array(x)), 
                ])
    target_transform = transforms.Compose([
                            transforms.Lambda(lambda x: np.array(x)), 
                ])
    train_data = datasets.CelebA(dataset_dir, split="train", transform=transform, target_transform=target_transform, download=True)
    val_data = datasets.CelebA(dataset_dir, split="valid", transform=transform, target_transform=target_transform, download=True)
    test_data = datasets.CelebA(dataset_dir, split="test", transform=transform, target_transform=target_transform, download=True)

    train_loader = DataLoader(train_data, num_workers=16, collate_fn=lambda x: x)
    val_loader = DataLoader(val_data, num_workers=16, collate_fn=lambda x: x)
    test_loader = DataLoader(test_data, num_workers=16, collate_fn=lambda x: x)

    for db, data_loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        lmdb_path = os.path.expanduser(os.path.join(outpath, 'celeba', f'{db}.db'))
        isdir = os.path.isdir(lmdb_path)

        print("Generate LMDB to %s" % lmdb_path)
        db = lmdb.open(lmdb_path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True)

        txn = db.begin(write=True)
        for idx, data in enumerate(data_loader):
            image, label = data[0]
            txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label)))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (idx, len(data_loader)))
                txn.commit()
                txn = db.begin(write=True)

        # finish iterating through dataset
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
            txn.put(b'__len__', pickle.dumps(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to original image dataset folder")
    parser.add_argument("-o", "--outpath", help="Path to output LMDB file")
    args = parser.parse_args()
    celeba2lmdb(args.dataset, args.outpath)