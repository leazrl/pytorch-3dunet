#type: ignore

import collections

collections.Sequence = collections.abc.Sequence
import os
import glob
from abc import abstractmethod

import h5py
import imageio.v2 as imageio
import numpy as np
from natsort import natsorted
from pathlib import Path
import skimage
import torch
from typing import Union, Optional, List


from pytorch3dunet.augment import transforms
from pytorch3dunet.datasets.hdf5 import _create_padded_indexes
from pytorch3dunet.datasets.utils import (
    ConfigDataset,
    calculate_stats,
    read_file_names,
    get_roi_slice,
    get_patch_size,
    get_slice_builder,
    mirror_pad,
)
from pytorch3dunet.unet3d.utils import get_logger, set_large_instances_to_zero

logger = get_logger("DSB2018Dataset")


def traverse_S_BIAD1410_paths(file_paths, find_masks=False):
    """
    Traverse the given list of file paths and include all non mask tif files found in the directories.
    """
    assert isinstance(file_paths, list), "file_paths should be a list of strings"
    results = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # find all files in directory with ending .tif or .h5 and not containing "mask"
            paths = glob.glob(
                os.path.join(file_path, "**/*.tif"), recursive=True
            ) + glob.glob(os.path.join(file_path, "**/*.h5"), recursive=True)
            if find_masks == True:
                condition = lambda x: "mask" in os.path.basename(x)
            else:
                condition = lambda x: "mask" not in os.path.basename(x)
            paths = list(filter(condition, paths))
            results.extend(paths)
        else:
            results.append(file_path)
    return sorted(results)


def dsb_prediction_collate(batch):
    """
    Forms a mini-batch of (images, paths) during test time for the DSB-like datasets.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], collections.Sequence):
        # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
        transposed = zip(*batch)
        return [dsb_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DSB2018Dataset(ConfigDataset):
    def __init__(
        self,
        root_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
    ):
        assert os.path.isdir(root_dir), f"{root_dir} is not a directory"
        assert phase in ["train", "val", "test"]

        self.phase = phase

        # load raw images
        images_dir = os.path.join(root_dir, "images")
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, expand_dims)
        self.file_path = images_dir

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            masks_dir = os.path.join(root_dir, "masks")
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(masks_dir, expand_dims)
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase != "test":
            mask = self.masks[idx]
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            return self.raw_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        file_paths = phase_config["file_paths"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [
            cls(
                file_paths[0],
                phase,
                transformer_config,
                expand_dims,
                dataset_config.get("global_norm", False),
                dataset_config.get("percentiles", None),
            )
        ]

    @staticmethod
    def _load_files(dir, expand_dims):
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
            path = os.path.join(dir, file)
            img = np.asarray(imageio.imread(path))
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths



class S_BIAD1410_Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to tif file containing raw data as well as labels and per pixel weights (optional)
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        slice_builder_config (dict): configuration of the SliceBuilder
        transformer_config (dict): data augmentation configuration
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        weight_internal_path (str or list): H5 internal path to the per pixel weights (optional)
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(
        self,
        img_path: str,
        mask_path:str,
        roi:Optional[List[List[int]]],
        phase:str,
        transformer_config,
        slice_builder_config=None,
        global_normalization=True,
        global_percentiles=None,
        image_key: Optional[str] = "predictions",
        mask_key: Optional[str] = None,
    ):
        assert phase in ["train", "val", "test", "eval"]

        self.phase = phase
        self.file_path = img_path
        self.label_file_path = mask_path
        if roi is not None:
            self.roi = get_roi_slice(roi)
        else:
            self.roi = roi

        if global_normalization:
            logger.info("Calculating mean and std of the raw data...")
            self.raw = self.load_data(self.file_path, image_key)
            if global_percentiles is not None:
                stats = calculate_stats(
                    self.raw,
                    percentile_min=global_percentiles[0],
                    percentile_max=global_percentiles[1],
                )
            else:
                stats = calculate_stats(self.raw)
        else:
            stats = calculate_stats(None, True)
            self.raw = self.load_data(self.file_path, image_key)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != "test":
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
            self.label = self.load_data(self.label_file_path, mask_key)
            self._check_volume_sizes()
            if phase == "eval":
                with h5py.File(self.file_path, "r") as f:
                    self.patch_indexes = f["patch_index"][:]
                self.patch_shape = get_patch_size(self.patch_indexes[0])
                self.patch_count = len(self.patch_indexes)
                logger.info(f"Number of patches: {self.patch_count}")

        if phase != "eval":
            self.patch_shape = slice_builder_config.get("patch_shape")
            self.halo_shape = slice_builder_config.get("halo_shape", [0, 0, 0])

            if phase == "test":
                # 'test' phase used only for predictions so ignore the label dataset
                self.label = None

                # compare patch and stride configuration
                patch_shape = slice_builder_config.get("patch_shape")
                stride_shape = slice_builder_config.get("stride_shape")
                if sum(self.halo_shape) != 0 and patch_shape != stride_shape:
                    logger.warning(
                        f"Found non-zero halo shape {self.halo_shape}. "
                        f"In this case: patch shape and stride shape should be equal for optimal prediction "
                        f"performance, but found patch_shape: {patch_shape} and stride_shape: {stride_shape}!"
                    )

            # build slice indices for raw and label data sets
            slice_builder = get_slice_builder(
                self.raw, self.label, None, slice_builder_config
            )
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices

            self.patch_count = len(self.raw_slices)
            logger.info(f"Number of patches: {self.patch_count}")

            self._raw_padded = None

    def get_raw_patch(self, idx):
        return self.raw[idx]

    def get_label_patch(self, idx):
        return self.label[idx]

    def get_raw_padded_patch(self, idx):
        if self._raw_padded is None:
            self._raw_padded = mirror_pad(self.raw, self.halo_shape)
        return self._raw_padded[idx]
    
    def load_data(self, path: Union[str, Path], key:Optional[str]):
        if path.endswith((".h5", ".hdf5")):
            with h5py.File(path, "r") as f:
                if self.roi is not None:
                    raw = f[key][self.roi]
                else:
                    raw = f[key][:]
        elif path.endswith(".tif"):
            raw = imageio.volread(path)
            if self.roi is not None:
                raw = self.raw[self.roi]
        return raw

    def volume_shape(self):
        raw = imageio.volread(self.file_path)
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        if self.phase == "eval":
            raw_patch_transformed = self.raw_transform(self.get_raw_patch(idx))
            if self.label.shape == self.raw.shape:
                # if the label shape is equal to the raw shape, then  should be a patchwise evaluation
                # and can use direct patch index
                label_idx = idx
            else:
                label_idx = get_roi_slice(self.patch_indexes[idx])
            label_patch_transformed = self.label_transform(
                self.get_label_patch(label_idx)
            )
            return raw_patch_transformed, label_patch_transformed

        else:
            raw_idx = self.raw_slices[idx]

            if self.phase == "test":
                if len(raw_idx) == 4:
                    # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                    raw_idx = raw_idx[
                        1:
                    ]  # Remove the first element if raw_idx has 4 elements
                    raw_idx_padded = (slice(None),) + _create_padded_indexes(
                        raw_idx, self.halo_shape
                    )
                else:
                    raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

                raw_patch_transformed = self.raw_transform(
                    self.get_raw_padded_patch(raw_idx_padded)
                )
                return raw_patch_transformed, raw_idx
            else:
                raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))

                # get the slice for a given index 'idx'
                label_idx = self.label_slices[idx]
                label_patch_transformed = self.label_transform(
                    self.get_label_patch(label_idx)
                )
                # return the transformed raw and label patches
                return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    def _check_volume_sizes(self):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        if self.file_path.endswith((".h5", ".hdf5")):
            with h5py.File(self.file_path, "r") as f:
                raw = f["predictions"][:]
        elif self.file_path.endswith(".tif"):
            raw = imageio.volread(self.file_path)
        label = imageio.volread(self.label_file_path)
        if self.phase == "eval":
            assert raw.ndim in [
                4,
                5,
            ], "Raw dataset must be 4D (NxCxHxW) or 5D (NxCxDxHxW)"
        else:
            assert raw.ndim in [3, 4], "Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)"
            assert _volume_shape(raw) == _volume_shape(
                label
            ), "Raw and labels have to be of the same size"
        assert label.ndim in [3, 4], "Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)"

    def get_patch_shape(self):
        return self.patch_shape

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load slice builder config
        slice_builder_config = phase_config["slice_builder"]
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        img_paths = traverse_S_BIAD1410_paths(phase_config["img_paths"])
        mask_paths = traverse_S_BIAD1410_paths(
            phase_config["mask_paths"], find_masks=True
        )

        roi = phase_config.get("roi", None)

        datasets = []
        for i, img_path in enumerate(img_paths):
            try:
                """
                if phase == "eval":
                    assert (
                        os.path.basename("_".join(img_path.split("_")[:-1]))
                        in mask_paths[i]
                    ), f"Image {img_path} does not have a corresponding mask in {mask_paths[i]}"
                else:
                    assert (
                        os.path.basename(img_path) in mask_paths[i]
                    ), f"Image {img_path} does not have a corresponding mask in {mask_paths[i]}"
                """
                logger.info(f"Loading {phase} set from: {img_path}...")
                dataset = cls(
                    img_path=img_path,
                    mask_path=mask_paths[i],
                    roi=roi,
                    phase=phase,
                    transformer_config=transformer_config,
                    slice_builder_config=slice_builder_config,
                    global_normalization=dataset_config.get(
                        "global_normalization", None
                    ),
                    global_percentiles=dataset_config.get("global_percentiles", None),
                    image_key=dataset_config.get("image_key", "predictions"),
                    mask_key=dataset_config.get("mask_key", None),
                )
                datasets.append(dataset)
            except Exception:
                logger.error(f"Skipping {phase} set: {img_path}", exc_info=True)
        return datasets


### General Datasets


class Abstract_TIF_Dataset(ConfigDataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        filenames_path=None,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
        image_key="predictions",
        mask_key=None,
        prediction_channel=None,
        min_object_size=None,
        instance_zero_background=False,
    ):
        assert os.path.isdir(image_dir), f"{image_dir} is not a directory"
        assert os.path.isdir(mask_dir), f"{mask_dir} is not a directory"
        assert phase in ["train", "val", "test", "eval"]

        self.phase = phase

        # load raw images
        assert os.path.isdir(image_dir)

        if filenames_path is not None:
            self.file_names = read_file_names(filenames_path)

        self.images, self.paths = self._load_files(image_dir, expand_dims, image_key, prediction_channel)
        self.file_path = image_dir

        if percentiles is None:
            percentile_min = None
            percentile_max = None
        else:
            percentile_min = percentiles[0]
            percentile_max = percentiles[1]
        if global_norm:
            stats = calculate_stats(
                self.images,
                False,
                percentile_min,
                percentile_max,
            )
        else:
            stats = calculate_stats(
                self.images,
                True,
                percentile_min,
                percentile_max,
            )

        transformer = transforms.Transformer(transformer_config, stats)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()

        if phase != "test":
            # load labeled images
            assert os.path.isdir(mask_dir)
            self.masks, _ = self._load_files(mask_dir, expand_dims, mask_key)
            assert len(self.images) == len(self.masks)
            # load label images transformer
            self.masks_transform = transformer.label_transform()
        else:
            self.masks = None
            self.masks_transform = None
        
        self.min_obj_size = min_object_size
        self.image_key = image_key
        self.instance_zero_background = instance_zero_background

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        print(self.paths[idx])
        if (self.phase == "eval") and (self.min_obj_size is not None) and (self.image_key == "segmentation"):
            img = skimage.morphology.remove_small_objects(
                img, min_size=self.min_obj_size
            )
        if (self.phase == "eval") and (self.instance_zero_background==True):
            img = set_large_instances_to_zero(img)
        if self.phase != "test":
            mask = self.masks[idx]
            if self.min_obj_size is not None:
                mask = skimage.morphology.remove_small_objects(
                    mask, min_size=self.min_obj_size
                )
            return self.raw_transform(img), self.masks_transform(mask)
        else:
            return self.raw_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)
    
    def filter_by_foreground_ratio(self, fg_ratio_threshold):
        """
        Filter out samples with foreground ratio below the given threshold.
        """
        if self.masks is None:
            raise ValueError("Masks are not loaded, cannot filter by foreground ratio.")
        
        length_before = len(self.images)

        filtered_images = []
        filtered_masks = []
        filtered_paths = []
        
        for img, mask, path in zip(self.images, self.masks, self.paths):
            fg_ratio = np.sum(mask > 0) / mask.size
            if fg_ratio >= fg_ratio_threshold:
                filtered_images.append(img)
                filtered_masks.append(mask)
                filtered_paths.append(path)
        
        self.images = filtered_images
        self.masks = filtered_masks
        self.paths = filtered_paths

        logger.info(f'After filtering by foreground ratio > {fg_ratio_threshold}: number of patches reduced from {length_before} to {len(filtered_images)}.')

    def subsample_by_fraction(self, fraction):
        """
        Subsample the dataset by the given fraction.
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError("Fraction must be in the range (0, 1].")
        
        total_samples = len(self.images)
        subsample_size = int(total_samples * fraction)
        
        self.images = self.images[:subsample_size]
        if self.masks is not None:
            self.masks = self.masks[:subsample_size]
        self.paths = self.paths[:subsample_size]

        logger.info(f'After subsampling by fraction {fraction}: number of patches reduced from {total_samples} to {len(self.images)}.')

    @classmethod
    def prediction_collate(cls, batch):
        return dsb_prediction_collate(batch)

    @abstractmethod
    def create_datasets(cls, dataset_config, phase):
        pass

    @abstractmethod
    def _load_files(self, dir, expand_dims, key, prediction_channel=None):
        pass


class Standard_TIF_Dataset(Abstract_TIF_Dataset):
    """Dataset for tif files arranged in a file structure
    of multiple single image tifs located in a single
    file with image and mask files located in differnt folders.
    e.g DSB2018, S_BIAD895

    Args:
        Abstract_TIF_Dataset (_type_): _description_
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
        image_key="predictions",
        mask_key=None,
        prediction_channel=None,
        min_object_size=None,
        instance_zero_background=False,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            phase=phase,
            transformer_config=transformer_config,
            expand_dims=expand_dims,
            global_norm=global_norm,
            percentiles=percentiles,
            image_key=image_key,
            mask_key=mask_key,
            prediction_channel=prediction_channel,
            min_object_size=min_object_size,
            instance_zero_background=instance_zero_background,
        )

    def _load_files(self, dir, expand_dims, key, prediction_channel=None):
        files_data = []
        paths = []
        for file in natsorted(os.listdir(dir)):
            if not file.startswith("."):
                if "metric_summary" in file:
                    continue
                path = os.path.join(dir, file)
                if file.endswith((".tif", ".png")):
                    img = np.asarray(imageio.imread(path))
                # check if file ends in ['.h5', '.hdf5']
                elif file.endswith((".h5", ".hdf5")):
                    with h5py.File(path, "r") as f:
                        img = f[key][:]
                    if prediction_channel is not None:
                        img = img[prediction_channel]
                else:
                    continue
                if expand_dims:
                    dims = img.ndim
                    img = np.expand_dims(img, axis=0)
                    if dims == 3:
                        img = np.transpose(img, (3, 0, 1, 2))

                files_data.append(img)
                paths.append(path)

        return files_data, paths

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        image_paths = phase_config["image_dir"]
        mask_paths = phase_config["mask_dir"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [
            cls(
                image_dir=image_paths[0],
                mask_dir=mask_paths[0],
                phase=phase,
                transformer_config=transformer_config,
                expand_dims=expand_dims,
                global_norm=dataset_config.get("global_norm", False),
                percentiles=dataset_config.get("percentiles", None),
                image_key=dataset_config.get("image_key", "predictions"),
                mask_key=dataset_config.get("mask_key", None),
                prediction_channel=dataset_config.get("prediction_channel", None),
                min_object_size=dataset_config.get("min_object_size", None),
                instance_zero_background=dataset_config.get("instance_zero_background", False),
            )
        ]


class Hoechst_Dataset(Abstract_TIF_Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
        image_key="predictions",
        mask_key=None,
        prediction_channel=None,
        min_object_size=None,
        instance_zero_background=False,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            phase=phase,
            transformer_config=transformer_config,
            expand_dims=expand_dims,
            global_norm=global_norm,
            percentiles=percentiles,
            image_key=image_key,
            mask_key=mask_key,
            prediction_channel=prediction_channel,
            min_object_size=min_object_size,
            instance_zero_background=instance_zero_background,
        )

    def _load_files(self, dir, expand_dims, key, prediction_channel=None):
        files_data = []
        paths = []
        for file in natsorted(os.listdir(dir)):
            if "metric_summary" in file:
                    continue
            if not file.startswith("."):
                path = os.path.join(dir, file)
                if file.endswith((".tif", ".png")):
                    img = np.asarray(imageio.imread(path))
                elif file.endswith((".h5", ".hdf5")):
                    with h5py.File(path, "r") as f:
                        img = f[key][:]
                    if prediction_channel is not None:
                        img = img[prediction_channel]
                if img.ndim == 3:
                    img = transforms.RgbToLabel()(img)
                if expand_dims:
                    dims = img.ndim
                    img = np.expand_dims(img, axis=0)
                    if dims == 3:
                        img = np.transpose(img, (3, 0, 1, 2))

                files_data.append(img)
                paths.append(path)

        return files_data, paths

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        image_paths = phase_config["image_dir"]
        mask_paths = phase_config["mask_dir"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [
            cls(
                image_dir=image_paths[0],
                mask_dir=mask_paths[0],
                phase=phase,
                transformer_config=transformer_config,
                expand_dims=expand_dims,
                global_norm=dataset_config.get("global_norm", False),
                percentiles=dataset_config.get("percentiles", None),
                image_key=dataset_config.get("image_key", "predictions"),
                mask_key=dataset_config.get("mask_key", None),
                prediction_channel=dataset_config.get("prediction_channel", None),
                min_object_size=dataset_config.get("min_object_size", None),
                instance_zero_background=dataset_config.get("instance_zero_background", False),
            )
        ]


class HeLaNuc_Dataset(Abstract_TIF_Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
        image_key="predictions",
        mask_key=None,
        prediction_channel=None,
        min_object_size=None,
        instance_zero_background=False,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            phase=phase,
            transformer_config=transformer_config,
            expand_dims=expand_dims,
            global_norm=global_norm,
            percentiles=percentiles,
            image_key=image_key,
            mask_key=mask_key,
            prediction_channel=prediction_channel,
            min_object_size=min_object_size,
            instance_zero_background=instance_zero_background,
        )

    def _load_files(self, dir, expand_dims, key, prediction_channel=None):
        files_data = []
        paths = []
        for file in natsorted(os.listdir(dir)):
            if "metric_summary" in file:
                    continue
            if not file.startswith("."):
                path = os.path.join(dir, file)
                if file.endswith((".tif", ".png")):
                    img = np.asarray(imageio.imread(path))
                elif file.endswith((".h5", ".hdf5")):
                    with h5py.File(path, "r") as f:
                        img = f[key][:]
                    if prediction_channel is not None:
                        img = img[prediction_channel]
                if img.ndim == 3:
                    # select last channel corresponding to nuclei channel
                    img = img[:, :, 2]
                if expand_dims:
                    dims = img.ndim
                    img = np.expand_dims(img, axis=0)
                    if dims == 3:
                        img = np.transpose(img, (3, 0, 1, 2))

                files_data.append(img)
                paths.append(path)

        return files_data, paths

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        image_paths = phase_config["image_dir"]
        mask_paths = phase_config["mask_dir"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [
            cls(
                image_dir=image_paths[0],
                mask_dir=mask_paths[0],
                phase=phase,
                transformer_config=transformer_config,
                expand_dims=expand_dims,
                global_norm=dataset_config.get("global_norm", False),
                percentiles=dataset_config.get("percentiles", None),
                image_key=dataset_config.get("image_key", "predictions"),
                mask_key=dataset_config.get("mask_key", None),
                prediction_channel=dataset_config.get("prediction_channel", None),
                min_object_size=dataset_config.get("min_object_size", None),
                instance_zero_background=dataset_config.get("instance_zero_background", False),
            )
        ]


class TIF_txt_Dataset(Abstract_TIF_Dataset):
    """Dataset for tif files arranged in a file structure
    of multiple single image tifs located in a single
    file with image and mask files located in differnt folders.
    With a txt file containing the filenames of images split
    into train, val and test sets.
    e.g BBBC039, S_BIAD634

    Args:
        Abstract_TIF_Dataset (_type_): _description_
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        phase,
        transformer_config,
        filenames_path,
        expand_dims=True,
        global_norm=False,
        percentiles=None,
        image_key="predictions",
        mask_key=None,
        prediction_channel=None,
        min_object_size=None,
        instance_zero_background=False,
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            phase=phase,
            transformer_config=transformer_config,
            filenames_path=filenames_path,
            expand_dims=expand_dims,
            global_norm=global_norm,
            percentiles=percentiles,
            image_key=image_key,
            mask_key=mask_key,
            prediction_channel=prediction_channel,
            min_object_size=min_object_size,
            instance_zero_background=instance_zero_background,
        )

    def _load_files(self, dir, expand_dims, key, prediction_channel=None):
        files_data = []
        paths = []
        for file in self.file_names:
            path = glob.glob(dir + f"/*{file}*")[0]
            if path.endswith((".tif", ".png")):
                img = np.asarray(imageio.imread(path))
            elif path.endswith((".h5", ".hdf5")):
                with h5py.File(path, "r") as f:
                    img = f[key][:]
                if prediction_channel is not None:
                    img = img[prediction_channel]
            if img.ndim == 3:
                img = img[:, :, 0]
            if expand_dims:
                dims = img.ndim
                img = np.expand_dims(img, axis=0)
                if dims == 3:
                    img = np.transpose(img, (3, 0, 1, 2))

            files_data.append(img)
            paths.append(path)

        return files_data, paths

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load files to process
        image_paths = phase_config["image_dir"]
        mask_paths = phase_config["mask_dir"]
        expand_dims = dataset_config.get("expand_dims", True)
        return [
            cls(
                image_dir=image_paths[0],
                mask_dir=mask_paths[0],
                phase=phase,
                transformer_config=transformer_config,
                filenames_path=phase_config.get("filenames_path", None),
                expand_dims=expand_dims,
                global_norm=dataset_config.get("global_norm", False),
                percentiles=dataset_config.get("percentiles", None),
                image_key=dataset_config.get("image_key", "predictions"),
                mask_key=dataset_config.get("mask_key", None),
                prediction_channel=dataset_config.get("prediction_channel", None),
                min_object_size=dataset_config.get("min_object_size", None),
                instance_zero_background=dataset_config.get("instance_zero_background", False),
            )
        ]
