# datasets/custom_hdf5_regression.py

import os
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.utils.continual_dataset import ContinualDataset, store_domain_loaders
from backbone.ResNet18 import resnet18
import torch.nn as nn

from utils.conf import base_data_path
from datasets.transforms.denormalization import DeNormalize  # for API symmetry with DomainNet

# Your 9 regression outputs, in the order used when building HDF5
LABEL_COLS = [
    'Vaccum Cleaning',
    'Mopping the Floor',
    'Carry Warm Food',
    'Carry Cold Food',
    'Carry Drinks',
    'Carry Small Objects',
    'Carry Large Objects',
    'Cleaning',
    'Starting a conversation'
]


class HDF5CLDataset(Dataset):
    """
    HDF5-backed dataset for a single (domain, split).
    Assumes HDF5 was created by create_domain_stratified_hdf5.
    Returns: (img, labels_9d, not_aug_img)
    """
    def __init__(self, hdf5_path, domain, split,
                 img_variant="image_path",
                 transform=None,
                 not_aug_transform=None):
        self.hdf5_path = hdf5_path
        self.domain = domain
        self.split = split
        self.img_variant = img_variant
        self.transform = transform
        self.not_aug_transform = not_aug_transform

        self._f = None
        self._length = None

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        if self._length is None:
            with h5py.File(self.hdf5_path, "r") as f:
                grp = f[f"{self.domain}/{self.split}"]
                self._length = len(grp["labels"])
        return self._length

    def __getitem__(self, idx):
        self._ensure_open()
        dom_split = f"{self.domain}/{self.split}"
        grp = self._f[dom_split]
        imgs_grp = grp["images"]
        labels_ds = grp["labels"]

        # images saved by your pipeline: (C, H, W) float32, already resized+normalized
        img_np = imgs_grp[self.img_variant][idx]
        img_tensor = torch.from_numpy(img_np).float()

        # no extra transforms: keep tensors as-is
        img = img_tensor
        not_aug_img = img_tensor

        # hooks kept for API symmetry; normally transform is None
        if self.transform is not None:
            img = self.transform(img_tensor)
        if self.not_aug_transform is not None:
            not_aug_img = self.not_aug_transform(img_tensor)

        # labels already scaled (raw-1)/4, shape (9,)
        labels_np = labels_ds[idx]
        labels = torch.tensor(labels_np, dtype=torch.float32)

        if self.split == "test":
            return img, labels
        else:
            return img, labels, not_aug_img


class CustomHDF5Regression(ContinualDataset):
    """
    Domain-incremental regression dataset backed by a single HDF5,
    structured analogously to DomainNet but with 9-d regression targets.
    """
    NAME = "custom-hdf5-regression"
    SETTING = "domain-2il"
    N_CLASSES_PER_TASK = 1
    N_TASKS = 6
    # IMG_SIZE = 64
    MEAN = [0.485, 0.456, 0.406] #only to mirror DomainNet's API
    STD = [0.229, 0.224, 0.225] #only to mirror DomainNet's API
    DOMAIN_LST = ['Home', 'BigOffice-2', 'BigOffice-3',
                  'Hallway', 'MeetingRoom', 'SmallOffice']
    
    # resize_64 = transforms.Resize((IMG_SIZE, IMG_SIZE))
    # TRANSFORM = [resize_64]
    # TRANSFORM_NORM = [resize_64]
    # TRANSFORM_TEST = [resize_64]
    # NOT_AUG_TRANSFORM = [resize_64]
    TRANSFORM = []
    TRANSFORM_NORM = []
    TRANSFORM_TEST = []
    NOT_AUG_TRANSFORM = []

    data_path = base_data_path()  # adjust folder if needed
    hdf5_path = os.path.join(data_path, "mean_data_pepper_fold0.hdf5")


    def get_data_loaders(self, task_id=None):
        # DomainNet appends normalize when args.aug_norm; we keep transforms empty
        if self.args.aug_norm:
            transform = transforms.Compose(self.TRANSFORM_NORM) if self.TRANSFORM_NORM else None
            test_transform = transforms.Compose(self.TRANSFORM_NORM) if self.TRANSFORM_NORM else None
        else:
            transform = transforms.Compose(self.TRANSFORM) if self.TRANSFORM else None
            test_transform = transforms.Compose(self.TRANSFORM_TEST) if self.TRANSFORM_TEST else None

        not_aug_transform = transforms.Compose(self.NOT_AUG_TRANSFORM) if self.NOT_AUG_TRANSFORM else None

        current_domain = self.DOMAIN_LST[self.i]

        train_dataset = HDF5CLDataset(
            hdf5_path=self.hdf5_path,
            domain=current_domain,
            split="train",
            img_variant="image_path",
            transform=transform,
            not_aug_transform=not_aug_transform
        )

        test_dataset = HDF5CLDataset(
            hdf5_path=self.hdf5_path,
            domain=current_domain,
            split="test",
            img_variant="image_path",
            transform=test_transform,
            not_aug_transform=None
        )

        train_loader, test_loader = store_domain_loaders(train_dataset, test_dataset, self)
        return train_loader, test_loader

    def not_aug_dataloader(self, batch_size):
        # Optional; not used by most methods
        pass

    @staticmethod
    def get_backbone(num_classifier=1, norm_feature=False, diff_classifier=False, num_rot=0, ema_classifier=False,
                     lln=False, dist_linear=False, algorithm='None', pretrained=False):
        return resnet18(9, norm_feature=norm_feature, diff_classifier=diff_classifier,
                            num_rot=num_rot, ema_classifier=ema_classifier, lln=lln, algorithm=algorithm,
                            pretrained=pretrained)

    @staticmethod
    def get_transform():
        # Identity: data already normalized in HDF5
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_norm_transform():
        # Also identity to avoid double-normalization
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_normalization_transform():
        # Kept for API completeness; no-op in your setup
        return transforms.Lambda(lambda x: x)

    @staticmethod
    def get_loss(use_bce=False):
        # Regression loss
        return nn.MSELoss(reduction="mean")

    @staticmethod
    def get_denormalization_transform():
        # For visualisation if needed; invert ImageNet norm
        return DeNormalize(CustomHDF5Regression.MEAN, CustomHDF5Regression.STD)
