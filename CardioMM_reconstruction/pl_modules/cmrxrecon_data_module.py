"""
Pytorch Lightning module to handle fastMRI and CMRxRecon data. 
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/pl_modules/data_module.py
Created on 2025/04/01
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.

Some codes are modified based on https://arxiv.org/abs/2309.13839
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch

import fastmri
from data.mri_data import CombinedCmrxReconSliceDataset, CmrxReconSliceDataset


def worker_init_fn(worker_id):
    """
    Seed each DataLoader worker's mask function RNG.

    PyTorch creates a different base seed for each worker. This helper forwards
    that seed to the dataset transform's ``mask_func`` RNG, with extra offsets
    for combined datasets and distributed training so workers/ranks do not share
    identical undersampling masks.

    Args:
        worker_id: Integer worker id provided by ``torch.utils.data.DataLoader``.
    """
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        CmrxReconSliceDataset, CombinedCmrxReconSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP so each rank can receive a unique seed stream.
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # For NumPy-compatible RNG seeding, final seeds must be within uint32 range.
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedCmrxReconSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                # Modulo keeps the seed accepted by NumPy RandomState.
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        # Modulo keeps the seed accepted by NumPy RandomState.
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    """
    Check whether two mutually exclusive sampling options are both set.

    This helper is used to reject configurations that specify both slice-level
    sampling and volume-level sampling for the same split.

    Args:
        val1: First optional value.
        val2: Second optional value.

    Returns:
        ``True`` when both values are not ``None``; otherwise ``False``.
    """
    if (val1 is not None) and (val2 is not None):
        return True

    return False


class CmrxReconDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CMRxRecon-style fastMRI datasets.

    This class owns split-specific transforms, optional slice/volume sampling,
    metadata cache warm-up, DataLoader construction, and distributed sampler
    setup. The configured root is derived from ``data_path / "Cine" /
    "TrainingSet" / h5py_folder`` and passed to the CMRxRecon slice datasets.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        h5py_folder: str,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            h5py_folder: Folder name under ``Cine/TrainingSet`` containing the
                converted HDF5 files.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            sample_rate: Fraction of slices of the training data split to use.
                Can be set to less than 1.0 for rapid prototyping. If not set,
                it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            val_sample_rate: Same as sample_rate, but for val split.
            test_sample_rate: Same as sample_rate, but for test split.
            volume_sample_rate: Fraction of volumes of the training data split
                to use. Can be set to less than 1.0 for rapid prototyping. If
                not set, it defaults to 1.0. To subsample the dataset either
                set sample_rate (sample by slice) or volume_sample_rate (sample
                by volume), but not both.
            val_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            test_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            train_filter: A callable which takes as input a training example
                metadata, and returns whether it should be part of the training
                dataset.
            val_filter: Same as train_filter, but for val split.
            test_filter: Same as train_filter, but for test split.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        # Each split may sample by slices or by volumes, but not both.
        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )
        if _check_both_not_none(test_sample_rate, test_volume_sample_rate):
            raise ValueError(
                "Can set test_sample_rate or test_volume_sample_rate, but not both."
            )
        # TODO: Easy working code, other mapping/aorta/mapping/... data will be loaded in CmrxReconSliceDataset
        # Current CMRxRecon root convention starts from the Cine training HDF5 folder.
        self.data_path = data_path / 'Cine' / 'TrainingSet' / h5py_folder  # Need to be checked sometimes
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for a train, validation, or test partition.

        Args:
            data_transform: Transform applied by the dataset for this split.
            data_partition: Split name, usually ``"train"``, ``"val"``,
                ``"test"``, or ``"challenge"``.
            sample_rate: Optional override for slice-level sampling.
            volume_sample_rate: Optional override for volume-level sampling.

        Returns:
            PyTorch DataLoader wrapping either ``CmrxReconSliceDataset`` or
            ``CombinedCmrxReconSliceDataset``.
        """
        if data_partition == "train":
            is_train = True
            # Training uses train-specific sampling and raw-sample filters.
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                # Validation uses its own sampling/filter configuration.
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter
            elif data_partition == "test":
                # Test uses its own sampling/filter configuration.
                sample_rate = (
                    self.test_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.test_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.test_filter

        # If desired, combine train and val together for leaderboard-style training.
        dataset: Union[CmrxReconSliceDataset, CombinedCmrxReconSliceDataset]
        if is_train and self.combine_train_val:  # TODO: Dataset for training !!! Need to be modified if needed !!!
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
            ]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedCmrxReconSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                # Explicit test_path overrides the split-derived path.
                data_path = self.test_path
            else:  # TODO: Dataset for validation !!! Need to be modified if needed !!!
                data_path = self.data_path / data_partition #"val" 

            dataset = CmrxReconSliceDataset(
                root=data_path,
                transform=data_transform,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                challenge=self.challenge,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
            )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                # VolumeSampler keeps full volumes on the same process during eval.
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers= self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            # When a sampler is provided, it controls ordering, so DataLoader shuffle is disabled.
            shuffle=is_train if sampler is None else False,
        )

        return dataloader
    
    def prepare_data(self):
        """
        Warm up dataset metadata caches before distributed training starts.

        Lightning calls this hook on the rank-zero process. Instantiating each
        split dataset once is enough to populate the cache file when dataset
        caching is enabled; no samples are consumed here.
        """
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / "val" #None #self.data_path #/ f"{self.challenge}_test"
            data_paths = [
                self.data_path / "train",# / f"{self.challenge}_train",
                self.data_path / "val", # / f"{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                # Dataset construction triggers cache generation/loading as a side effect.
                _ = CmrxReconSliceDataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                )

    def train_dataloader(self):
        """Return the training DataLoader with the training transform."""
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        """Return the validation DataLoader with the validation transform."""
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        """Return the test/challenge DataLoader with the test transform."""
        return self._create_data_loader(
            self.test_transform, data_partition=self.test_split
        )

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define CLI arguments for data paths, sampling, caching, and DataLoaders.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments: root paths, split selection, and sampling controls
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--h5py_folder",
            default=None,
            type=str,
            help="folder name for converted h5py files",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="singlecoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--test_split",
            choices=("val", "test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--val_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--test_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (test split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with sample_rate."
            ),
        )
        parser.add_argument(
            "--val_volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with val_sample_rate."
            ),
        )
        parser.add_argument(
            "--test_volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (test split only). If not "
                "given all will be used. Cannot set together with test_sample_rate."
            ),
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            action="store_true",
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments: batch size and worker count
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser
