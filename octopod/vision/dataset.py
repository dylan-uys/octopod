import os
import numpy as np
from PIL import Image
from io import BytesIO
import multiprocessing as mp
from sklearn import preprocessing
import ctypes
import torch
import boto3
import zarr
import time
from torch.utils.data import Dataset
from octopod.vision.config import cropped_transforms, full_img_transforms
from octopod.vision.helpers import center_crop_pil_image


def _create_shared_array(n_elems):
    return np.ctypeslib.as_array(mp.Array(ctypes.c_float, n_elems).get_obj())


class OctopodImageDataset(Dataset):
    """
    Load data specifically for use with a image models

    Parameters
    ----------
    x: pandas Series
        file paths to stored images
    y: list
        A list of dummy-encoded categories or strings
        For instance, y might be [0,1,2,0] for a 3 class problem with 4 samples,
        or strings which will be encoded using a sklearn label encoder
    transform: str or list of PyTorch transforms
        specifies how to preprocess the full image for a Octopod image model
        To use the built-in Octopod image transforms, use the strings: `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    crop_transform: str or list of PyTorch transforms
        specifies how to preprocess the center cropped image for a Octopod image model
        To use the built-in Octopod image transforms, use strings `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    """
    def __init__(self,
                 x,
                 y,
                 s3_bucket=None,
                 use_cropped_image=True,
                 transform='train',
                 crop_transform='train',
                 cache_dir='image_vectors'):
        self.x = x
        self.y = y

        self.x_cache_state = _create_shared_array(len(x))
        self.x_cache_state = np.zeros(len(x))
        self.x_cropped_cache_state = _create_shared_array(len(x))
        self.x_cropped_cache_state = np.zeros(len(x))

        self.cache_dir = cache_dir
        self.use_cropped_image = use_cropped_image
        self.s3_bucket = s3_bucket
        self.s3_client = None if self.s3_bucket is None else boto3.client('s3')
        self.label_encoder, self.label_mapping = self._encode_labels()

        os.makedirs(self.cache_dir, exist_ok=True)

        if transform in ('train', 'val'):
            self.transform = full_img_transforms[transform]
        else:
            self.transform = transform

        if crop_transform in ('train', 'val'):
            self.crop_transform = cropped_transforms[crop_transform]
        else:
            self.crop_transform = crop_transform

    def _cache_image(self, x, index, cache_state_array, suffix=''):
        """Write preprocessed image to zarr file, update cache_dict member variable
        to point to cached file"""
        target_fpath = os.path.join(self.cache_dir, f'{index}{suffix}.zarr')
        zarr.save(target_fpath, x.numpy())
        cache_state_array[index] = 1

    def _load_cached_image(self, index, cache_state_array, suffix=''):
        target_fpath = os.path.join(self.cache_dir, f'{index}{suffix}.zarr')
        return torch.from_numpy(zarr.load(target_fpath)[:])

    def __getitem__(self, index):
        """Return tuple of images as PyTorch tensors and and tensor of labels"""
        label = self.y[index]
        label = self.label_encoder.transform([label])[0]
        label = torch.from_numpy(np.array(label)).long()

        # load and preprocess image
        if self.x_cache_state[index]:
            # if this image has already been preprocessed and cached, load its vector
            full_img = self._load_cached_image(index, self.x_cache_state)
        else:
            # otherwise, load the original image, preprocess it and cache it
            if self.s3_bucket is not None:
                file_byte_string = self.s3_client.get_object(
                    Bucket=self.s3_bucket, Key=self.x[index])['Body'].read()
                full_img = Image.open(BytesIO(file_byte_string)).convert('RGB')
            else:
                full_img = Image.open(self.x[index]).convert('RGB')

        if self.x_cropped_cache_state[index]:
            # process cropped image
            if index in self.x_cropped_cache_state:
                # if this image has already been preprocessed and cached, load its vector
                cropped_img = self._load_cached_image(index, self.x_cropped_cache_state, '_cropped')
            else:
                # otherwise, crop preprocess and cache
                cropped_img = center_crop_pil_image(full_img)
                cropped_img = self.crop_transform(cropped_img)
                self._cache_image(cropped_img, index, self.x_cropped_cache_state, '_cropped')

            full_img = self.transform(full_img)
            self._cache_image(full_img, index, self.x_cache_state)
            return {'full_img': full_img,
                    'crop_img': cropped_img}, label

        if not isinstance(full_img, torch.Tensor):
            full_img = self.transform(full_img)
            self._cache_image(full_img, index, self.x_cache_state)
        return {'full_img': full_img}, label

    def __len__(self):
        return len(self.x)

    def _encode_labels(self):
        """Encodes y labels using sklearn to create allow for string or numeric inputs"""
        le = preprocessing.LabelEncoder()
        le.fit(self.y)
        mapping_dict = dict(zip(le.transform(le.classes_), le.classes_))
        return le, mapping_dict


class OctopodImageDatasetMultiLabel(OctopodImageDataset):
    """
    Subclass of OctopodImageDataset used for multi-label tasks

    Parameters
    ----------
    x: pandas Series
        file paths to stored images
    y: list
        a list of lists of binary encoded categories or strings with length equal to number of
        classes in the multi-label task. For a 4 class multi-label task
        a sample list would be [1,0,0,1], A string example would be ['cat','dog'],
        (if the classes were ['cat','frog','rabbit','dog]), which will be encoded
        using a sklearn label encoder to [1,0,0,1].
    transform: str or list of PyTorch transforms
        specifies how to preprocess the full image for a Octopod image model
        To use the built-in Octopod image transforms, use the strings: `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    crop_transform: str or list of PyTorch transforms
        specifies how to preprocess the center cropped image for a Octopod image model
        To use the built-in Octopod image transforms, use strings `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    """

    def __getitem__(self, index):
        """Return tuple of images as PyTorch tensors and and tensor of labels"""
        label = self.y[index]
        label = list(self.label_encoder.transform([label])[0])
        full_img = Image.open(self.x[index]).convert('RGB')

        cropped_img = center_crop_pil_image(full_img)

        full_img = self.transform(full_img)
        cropped_img = self.crop_transform(cropped_img)

        label = torch.FloatTensor(label)

        return {'full_img': full_img,
                'crop_img': cropped_img}, label

    def _encode_labels(self):
        """Encodes y labels using sklearn to create allow for string or numeric inputs"""
        mlb = preprocessing.MultiLabelBinarizer()
        mlb.fit(self.y)
        mapping_dict = dict(zip(list(range(0, len(mlb.classes_))), mlb.classes_))

        return mlb, mapping_dict
