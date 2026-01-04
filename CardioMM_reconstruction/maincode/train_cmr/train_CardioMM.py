"""
Training the CardioMM model with CSM estimation using SEMnetwork - pytorch - ddp and multiple GPU - two lmhead for metadata and undersampling
- lightweight prompt
- using CLIP text encoder
- Modified KspaceACSExtractor for 1D and 2D undersampling
- no medical condition injection
- Modified Data consistency
- Modified Text information injection
- Modified Learning rate scheme
Created on 2025/09/27
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.

Some codes are modified based on https://arxiv.org/abs/2309.13839
"""

import os
import sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

from data.transforms_metausv3 import CardioMMDataTransform
from pl_modules.cmrxrecon_data_module import CmrxReconDataModule
from pl_modules.CardioMM_module import CardioMM_Module
from data.subsample import create_mask_for_mask_type
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from packaging import version


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, None, args.accelerations, args.center_numbers
    )
    # use equispaced_fixed masks for train transform, fixed masks for val transform
    train_transform = CardioMMDataTransform(mask_func=mask, llm_model_name=args.llm_model_name, use_seed=False)
    val_transform = CardioMMDataTransform(mask_func=mask, llm_model_name=args.llm_model_name)
    test_transform = CardioMMDataTransform()
    # ptl data module - this handles data loaders
    data_module = CmrxReconDataModule(
        data_path=args.data_path,
        h5py_folder=args.h5py_folder,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=args.combine_train_val,  # combine train and val data for train
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.nworkers,  # default=0
        distributed_sampler=(args.strategy in (
            "ddp_find_unused_parameters_false", "ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism
    model = CardioMM_Module(
        num_cascades=args.num_cascades,
        num_adj_slices=args.num_adj_slices,
        n_feat0=args.n_feat0,
        feature_dim=args.feature_dim,
        prompt_dim=args.prompt_dim,

        sens_n_feat0=args.sens_n_feat0,
        sens_feature_dim=args.sens_feature_dim,
        sens_prompt_dim=args.sens_prompt_dim,

        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,

        use_checkpoint=args.use_checkpoint,
        low_mem=args.low_mem,

        llm_model_dim=args.llm_model_dim,  # llm model dim, CLIP text encoder should be 512
        llm_embd_dim=args.llm_embd_dim,  # llm embadding dim
        llm_nclasses=args.llm_nclasses,  # no use in our reconstruction task, or can be equalled to the number of undersampling patterns
        llm_model_name=args.llm_model_name,  # pre-trained language model name
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    # num_gpus = 2
    backend = "ddp"
    # For multi-gpu with frozen llm: "ddp", which means find_unused_parameters=True
    batch_size = 1  # for each gpu

    # set defaults based on optional directory config
    data_path = pathlib.Path('.')
    default_root_dir = data_path / "experiments"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    parser.add_argument(
        "--num_gpus",
        default=2,
        type=int,
        help="Number of GPUs to use",
    )

    parser.add_argument(
        "--nworkers",
        default=0,
        type=int,
        help="Number of CPU workers to use",
    )

    parser.add_argument(
        "--exp_name",
        default="metausunetlight",
        type=str,
        help="experiment name",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random_fixed", "equispaced_fixed", "radial_fixed", "random_equispaced_fixed", "random_equispaced_radial_fixed"),
        default="equispaced_fixed",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_numbers",
        nargs="+",
        default=[24],
        type=int,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument(
        "--llm_model_name",
        default="../clip-vit-base-patch16",
        type=str,
        help="Pre-trained language model name",
    )

    # data config with path to fastMRI data and batch size
    parser = CmrxReconDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="equispaced_fraction",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # module+text config
    parser = CardioMM_Module.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=12,  # number of unrolled iterations
        num_adj_slices=1,  # number of adjacent slices

        n_feat0=48,  # number of top-level channels for CardioMM model
        feature_dim=[72, 96, 120],
        prompt_dim=[24, 48, 72],

        sens_n_feat0=24,
        sens_feature_dim=[36, 48, 60],
        sens_prompt_dim=[12, 24, 36],

        len_prompt=[3, 3, 3],
        prompt_size=[64, 32, 16],
        n_enc_cab=[2, 3, 3],
        n_dec_cab=[2, 2, 3],
        n_skip_cab=[1, 1, 1],
        n_bottleneck_cab=3,
        no_use_ca=False,

        lr=0.0002,  # AdamW learning rate;
        lr_step_size=5,  # epoch at which to decrease learning rate
        lr_gamma=0.3,  # extent to which to decrease learning rate
        weight_decay=1e-2,  # weight regularization strength
        use_checkpoint=False,  # use checkpointing for GPU memory savings

        # text module config
        llm_model_dim=512,  # llm model dim, CLIP text encoder should be 512
        llm_embd_dim=256,  # llm embadding dim
        llm_nclasses=3,  # no use in our reconstruction task, or can be equalled to the number of undersampling patterns
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        # gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        strategy=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=False,  # makes things slower, but deterministic
        # default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=15,  # max number of epochs
        gradient_clip_val=0.01
    )

    args = parser.parse_args()
    args.gpus = args.num_gpus  # override pl.Trainer gpus arg
    pattern_folder = args.mask_type
    acc_folder = "acc_" + "_".join(map(str, args.accelerations))
    args.default_root_dir = default_root_dir / args.h5py_folder / args.exp_name / pattern_folder / acc_folder

    # logger
    logger1 = TensorBoardLogger(save_dir=args.default_root_dir, name="lightning_logs", version=None)
    logger2 = CSVLogger(save_dir=args.default_root_dir, name="lightning_logs_csv", version=None)
    if version.parse(pl.__version__) >= version.parse("1.6.0"):
        args.logger = [logger1, logger2]
    else:
        args.logger = logger1

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=5,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print(os.getcwd())

    run_cli()
