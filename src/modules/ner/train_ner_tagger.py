#!/usr/bin/env python
# coding: utf-8

# # 0. Preparation

# ### 0-1. Dependencies

import argparse
import os
import random
from logging import DEBUG, INFO, FileHandler, Formatter, StreamHandler, getLogger

import numpy as np
import pytorch_lightning as pl
import torch
from ner_tagger import NERTagger


def main():
    config = get_args()
    print(config)

    # ### 1-1. Print config
    create_logger(config.version)
    get_logger(config.version).info(config)

    # ### 1-2. Determinism
    seed_everything(config.random_state)

    # ### 1-3. DataLoader -> BioELMo -> CRF
    tagger = NERTagger(config)

    # ### 1-4. Training
    if config.cuda is None:
        device = torch.device("cuda")
    else:
        device = torch.device(f"cuda:{config.cuda}")

    tagger.to(device)

    MODEL_CHECK_POINT_PATH = {
        "bioelmo": "./models/tagging_with_bioelmo_crf",
        "biobert": "./models/tagging_with_biobert_crf",
    }

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=f"{MODEL_CHECK_POINT_PATH[config.model]}/{config.version}.ckpt"
    )

    trainer = pl.Trainer(
        max_epochs=int(config.max_epochs),
        fast_dev_run=bool(config.debug_mode),
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(tagger)

    # ### 1-5. Test
    trainer.test()


# ### 0-1. Prepare for logging


def create_logger(exp_version):
    log_file = "{}.log".format(exp_version)

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)


# ### 0-2. Determinism


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ### 0-3. Hyperparameters


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--debug",
        "--debug-mode",
        action="store_true",
        dest="debug_mode",
        help="Set this option for debug mode",
    )
    parser.add_argument(
        "--train-dirs",
        dest="train_dirs",
        nargs="*",
        help="Directories of training dataset",
    )
    parser.add_argument(
        "--val-dirs", dest="val_dirs", nargs="*", help="Directories of validation dataset"
    )
    parser.add_argument(
        "--test-dirs", dest="test_dirs", nargs="*", help="Directories of test dataset"
    )
    parser.add_argument(
        "--model",
        default="bioelmo",
        dest="model",
        type=str,
        choices=["bioelmo", "biobert"],
        help="bioelmo or biobert",
    )
    parser.add_argument(
        "--bioelmo-dir",
        dest="bioelmo_dir",
        type=str,
        default="./models/bioelmo",
        help="BioELMo Directory",
    )
    parser.add_argument(
        "--biobert-path",
        dest="biobert_path",
        type=str,
        default="./models/biobert/biobert_v1.1_pubmed",
        help="BioBERT Directory",
    )
    parser.add_argument(
        "-v", "--version", dest="version", type=str, help="Experiment Name"
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default="15",
        help="Max Epochs (Default: 15)",
    )
    parser.add_argument(
        "--max-length",
        dest="max_length",
        type=int,
        default="1024",
        help="Max Length (Default: 1024)",
    )
    parser.add_argument(
        "-l",
        "--lr",
        dest="lr",
        type=float,
        default="1e-2",
        help="Learning Rate (Default: 1e-2)",
    )
    parser.add_argument(
        "--fine-tune-bioelmo",
        action="store_true",
        dest="fine_tune_bioelmo",
        help="Whether to Fine Tune BioELMo",
    )
    parser.add_argument(
        "--lr-bioelmo",
        dest="lr_bioelmo",
        type=float,
        default="1e-4",
        help="Learning Rate in BioELMo Fine-tuning",
    )
    parser.add_argument(
        "--fine-tune-biobert",
        action="store_true",
        dest="fine_tune_biobert",
        help="Whether to Fine Tune BioELMo",
    )
    parser.add_argument(
        "--lr-biobert",
        dest="lr_biobert",
        type=float,
        default="2e-5",
        help="Learning Rate in BioELMo Fine-tuning",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        default="16",
        help="Batch size (Default: 16)",
    )
    parser.add_argument(
        "-c", "--cuda", dest="cuda", default=None, help="CUDA Device Number"
    )
    parser.add_argument(
        "-r",
        "--random-state",
        dest="random_state",
        type=int,
        default="42",
        help="Random state (Default: 42)",
    )
    namespace = parser.parse_args()
    return namespace


if __name__ == "__main__":
    main()
