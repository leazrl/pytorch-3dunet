#type: ignore


import collections
from typing import Any, Optional, Union, List

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from pytorch3dunet.unet3d.utils import get_logger, get_class

logger = get_logger('Dataset')


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the patch and stride shape.

    Args:
        raw_dataset (ndarray): raw data
        label_dataset (ndarray): ground truth labels
        weight_dataset (ndarray): weights for the labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        kwargs: additional metadata
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs):
        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset, patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x),
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, ignore_index=None,
                 threshold=0.6, slack_acceptance=0.01, **kwargs):
        super().__init__(raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs)
        if label_dataset is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = np.copy(label_dataset[label_idx])
            if ignore_index is not None:
                for ii in ignore_index:
                    patch[patch == ii] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        logger.info(f'Filtering slices...')
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


    
class SingleZSliceBuilder(SliceBuilder):
    """
    Custom SliceBuilder that ensures exactly one patch per Z slice,
    centered in Y and X dimensions. Only returns patches where the patch_shape
    fits within the dataset dimensions.
    """
    
    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs):
        # Store original parameters
        self.patch_shape = tuple(patch_shape)
        self.stride_shape = tuple(stride_shape)
        
        # Don't call parent __init__ - we'll build slices ourselves
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)
        
        # Validate that patch fits in dataset dimensions
        self._validate_patch_fits(raw_dataset, patch_shape)
        
        # Build custom slices
        self._raw_slices = self._build_single_z_slices(raw_dataset, patch_shape, stride_shape)
        
        if label_dataset is None:
            self._label_slices = None
        else:
            self._validate_patch_fits(label_dataset, patch_shape)
            self._label_slices = self._build_single_z_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
            
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._validate_patch_fits(weight_dataset, patch_shape)
            self._weight_slices = self._build_single_z_slices(weight_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._weight_slices)

    def _validate_patch_fits(self, dataset, patch_shape):
        """Validate that the patch_shape fits within the dataset dimensions"""
        if dataset.ndim == 4:
            _, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape
        
        k_z, k_y, k_x = patch_shape
        
        if i_z < k_z:
            raise ValueError(f"Patch size Z dimension ({k_z}) is larger than dataset Z dimension ({i_z})")
        if i_y < k_y:
            raise ValueError(f"Patch size Y dimension ({k_y}) is larger than dataset Y dimension ({i_y})")
        if i_x < k_x:
            raise ValueError(f"Patch size X dimension ({k_x}) is larger than dataset X dimension ({i_x})")

    def _build_single_z_slices(self, dataset, patch_shape, stride_shape):
        """Build slices ensuring exactly one centered patch per Z slice"""
        slices = []
        
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape
        
        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        
        # For Z dimension: use the original logic to get all Z positions
        z_steps = list(self._gen_indices(i_z, k_z, s_z))
        
        # For Y and X dimensions: Use centered positions only
        # Calculate center positions for Y and X dimensions
        y_pos = (i_y - k_y) // 2
        x_pos = (i_x - k_x) // 2
        
        # Ensure positions are non-negative (should be guaranteed by validation)
        y_pos = max(0, y_pos)
        x_pos = max(0, x_pos)
        
        # Build slices: exactly one centered Y position and one centered X position per Z
        for z in z_steps:
            slice_idx = (
                slice(z, z + k_z),
                slice(y_pos, y_pos + k_y),
                slice(x_pos, x_pos + k_x),
            )
            if dataset.ndim == 4:
                slice_idx = (slice(0, in_channels),) + slice_idx
            slices.append(slice_idx)
        
        return slices



def _loader_classes(class_name):
    modules = [
        'pytorch3dunet.datasets.hdf5',
        'pytorch3dunet.datasets.dsb',
        'pytorch3dunet.datasets.utils',
        'domain_gap.dataset'
    ]
    return get_class(class_name, modules)


def get_slice_builder(raws, labels, weight_maps, config):
    assert 'name' in config
    logger.info(f"Slice builder config: {config}")
    slice_builder_cls = _loader_classes(config['name'])
    return slice_builder_cls(raws, labels, weight_maps, **config)


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

   #assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
    #    "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    train_datasets = dataset_class.create_datasets(loaders_config, phase='train')

    val_datasets = dataset_class.create_datasets(loaders_config, phase='val')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    batch_size_val = loaders_config.get('batch_size_val', batch_size)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train loader: {batch_size}')
    logger.info(f'Batch size for val loader: {batch_size_val}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, pin_memory=True,
                            num_workers=num_workers, shuffle=True)
    ### Read only for purposes of integrating with torch_em code, does not actually change dataloader shuffle status
    train_loader.shuffle = True
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size_val, pin_memory=True,
                          num_workers=num_workers, shuffle=False)
    ## Read only for purposes of integrating with torch_em code, does not actually change dataloader shuffle status
    val_loader.shuffle = False
    return {
        'train': train_loader,
        'val': val_loader
    }
    # return {
    #     'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True, pin_memory=True,
    #                         num_workers=num_workers),
    #     # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
    #     'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size_val, shuffle=False, pin_memory=True,
    #                       num_workers=num_workers)
    # }


def get_test_loaders(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase='test')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         collate_fn=collate_fn)
        
def get_filtered_test_loaders(config):
    """
    Returns test DataLoader for AdaBN.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders'].copy()

    logger.info('Creating test set loaders for AdaBN...')

    dataset_cls_str = loaders_config.get('dataset', None)

    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
        
    dataset_class = _loader_classes(dataset_cls_str) 
    test_datasets = dataset_class.create_datasets(loaders_config, phase='val')   # create datasets in phase 'val' to have access to labels for filtering

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()
    
    for ds in test_datasets:
        logger.info(f'Loading test set from: {ds.file_path}...')
        
        
        # get patches that are centered on nuclei before filtering for background

        patch_centroid = config.get('patch_centroid', False)
        allow_overlap = config.get('allow_overlap', True)

        if patch_centroid:
            ds.filter_by_centroids(overlap=allow_overlap)


        # filter by foreground ratio
        foreground_ratio_threshold = config.get('foreground_ratio_threshold', 0.0)
        if foreground_ratio_threshold > 0.0:
            logger.info(f'Filtering test set by foreground ratio threshold: {foreground_ratio_threshold}...')
            ds.filter_by_foreground_ratio(foreground_ratio_threshold)

        # subsample by amount of patches to use for AdaBN
        total_patches = len(ds)
        n_patches = config.get('n_patches', total_patches)
        if n_patches is not None:
            if n_patches > total_patches:
                logger.warning(
                    f'Requested n_patches={n_patches} exceeds available patches={total_patches}. '
                    f'Using all {total_patches} patches instead.'
                )
            elif n_patches < total_patches:
                logger.info(f'Subsampling test set from {total_patches} to {n_patches} patches...')
                ds.subsample_by_count(n_patches)
            else:
                logger.info(f'n_patches={n_patches} equals total patches, no subsampling needed.')
        
      
        # how to collate batches of data for prediction 
        if hasattr(ds, 'prediction_collate'):
            collate_fn = ds.prediction_collate
        else:
            collate_fn = default_prediction_collate 

        yield DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

def get_val_loader(loaders_config):
    """
    Returns dictionary containing the validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: val_loader
    """

    # get dataset class
    dataset_cls_str = loaders_config.get("dataset", None)
    if dataset_cls_str is None:
        dataset_cls_str = "StandardHDF5Dataset"
        logger.warning(
            f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'."
        )
    dataset_class = _loader_classes(dataset_cls_str)

    # assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
    #    "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    val_datasets = dataset_class.create_datasets(loaders_config, phase="val")

    num_workers = loaders_config.get("num_workers", 1)
    logger.info(f"Number of workers for val dataloader: {num_workers}")
    batch_size = loaders_config.get("batch_size", 1)

    logger.info(f"Batch size for val loader: {batch_size}")
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return [DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )]

def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def calculate_stats(
        img: Union[np.array, List[np.array]], 
        skip: bool = False, 
        percentile_min:Optional[float]=None, 
        percentile_max:Optional[float]=None,
    ) -> dict[str, Any]:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    # if img is list, flatten and combine items of list
    if isinstance(img, list):
        img = np.concatenate([np.ravel(arr) for arr in img])
    if not skip:
        mean = np.mean(img)
        std = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        if percentile_min is not None:
            pmin = np.percentile(img, percentile_min)
        else:
            pmin = None
        if percentile_max is not None:
            pmax = np.percentile(img, percentile_max)
        else:
            pmax = None

    else:
        pmin, pmax, mean, std, min_val, max_val = None, None, None, None, None, None
    

    return {
        'pmin': pmin,
        'pmax': pmax,
        'mean': mean,
        'std': std,
        'min_value': min_val,
        'max_value': max_val,
        'percentile_min': percentile_min,
        'percentile_max': percentile_max,
    }


def mirror_pad(image, padding_shape):
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert len(padding_shape) == 3, "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode='reflect')


def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]


def get_roi_slice(roi):
    # Create a tuple of slice objects based on the input list
    slices = tuple(slice(start, stop) for start, stop in roi)
    return slices

def get_patch_size(patch_index):
    patch_size = [0, 0, 0]
    for i, index in enumerate(patch_index):
        patch_size[i] = index[1] - index[0]
    return patch_size


def read_file_names(path):
    with open(path, 'r') as f:
        return f.read().splitlines()
    
