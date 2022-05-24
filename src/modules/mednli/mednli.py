import argparse
import os
import random
from logging import DEBUG, INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForPreTraining, BertModel, BertTokenizer

LABEL_TO_ID = {"neutral": 0, "contradiction": 1, "entailment": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def main():
    seed_everything()

    args = get_args()

    args.data_dir = "~/kart/biomedical_ner/corpus/mednli"

    if args.bert_path:
        PATH_BERT = Path(str(args.bert_path))
        bertconfig = BertConfig.from_pretrained("bert-base-uncased")
        bertforpretraining = BertForPreTraining(bertconfig)
        bertforpretraining.load_tf_weights(bertconfig, PATH_BERT)
        args.bert = bertforpretraining.bert
    else:
        args.bert = BertModel.from_pretrained("bert-base-uncased")

    create_logger(args.experiment_name)
    get_logger(args.experiment_name).info(str(dict(args._get_kwargs())))

    trainer = pl.Trainer(max_epochs=args.max_epoch)
    classifier = MedNliClassifier(args).to(torch.device(f"cuda:{args.cuda}"))
    trainer.fit(classifier)
    trainer.test()


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


class MedNliTsvDataset(Dataset):
    def __init__(self, tsv_path):
        super().__init__()
        self.df = pd.read_csv(tsv_path, sep="\t", header=None)
        self.df.columns = ["sentence_1", "sentence_2", "label"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix: int) -> Dict[str, str]:
        return {k: v.strip() for k, v in dict(self.df.iloc[ix, :]).items()}


class TextEncoder(torch.nn.Module):
    def __init__(self, bert: BertModel, tokenizer: BertTokenizer):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.hidden_to_score = torch.nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, batch: Dict[str, List[str]]):
        bert_input = self.get_bert_input(batch)

        # (n_batch, max_len, hidden_dim)
        bert_output = self.get_bert_output(batch)

        # suppress attention-masked token output to zero
        masked_bert_output = bert_output * bert_input["attention_mask"].unsqueeze(-1)

        # (n_batch, hidden_dim)
        averaged_masked_bert_output = masked_bert_output.sum(dim=1) / bert_input[
            "attention_mask"
        ].sum(dim=1).unsqueeze(-1)

        # (n_batch, 3)
        doc_score = self.hidden_to_score(averaged_masked_bert_output)

        # (n_batch, 3)
        doc_score_logsoftmax = torch.nn.functional.log_softmax(doc_score, dim=1)

        return doc_score_logsoftmax

    def get_bert_input(self, batch: Dict[str, List[str]]) -> Dict:
        sentence_pairs = [
            [s_1, s_2] for s_1, s_2 in zip(batch["sentence_1"], batch["sentence_2"])
        ]
        bert_input = self.tokenizer.batch_encode_plus(
            sentence_pairs, return_tensors="pt", pad_to_max_length=True
        )
        bert_input = {k: v.to(self.get_device()) for k, v in bert_input.items()}
        return bert_input

    def get_bert_output(self, batch: Dict[str, List[str]]) -> Dict:
        bert_input = self.get_bert_input(batch)

        # (n_batch, max_len, hidden_dim)
        bert_output = self.bert(**bert_input)[0]

        return bert_output

    def get_device(self):
        return list(self.state_dict().values())[0].device


class MedNliClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = TextEncoder(self.hparams.bert, self.tokenizer)
        self.lossfunc = torch.nn.NLLLoss()

    def get_device(self):
        return list(self.state_dict().values())[0].device

    def forward(self, batch) -> Dict[str, torch.tensor]:
        logits = self.encoder(batch)
        T = torch.tensor([LABEL_TO_ID[lbl] for lbl in batch["label"]]).to(
            self.get_device()
        )
        Y = logits.argmax(dim=1)
        loss = self.lossfunc(logits, T)
        return {"loss": loss, "T": T, "Y": Y}

    def calculate_score(self, answers, predictions) -> float:
        answers = answers.detach().cpu()
        predictions = predictions.detach().cpu()
        return accuracy_score(answers, predictions)

    def training_step(self, batch, batch_idx) -> Dict[str, torch.tensor]:
        return self.forward(batch)

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.tensor]:
        return self.forward(batch)

    def test_step(self, batch, batch_idx) -> Dict[str, torch.tensor]:
        return self.forward(batch)

    def epoch_end(self, outputs: List[Dict[str, torch.tensor]]):
        answers = torch.cat([output["T"] for output in outputs])
        predictions = torch.cat([output["Y"] for output in outputs])
        loss = torch.cat([output["loss"].unsqueeze(-1) for output in outputs]).sum()
        accuracy = self.calculate_score(answers, predictions)
        return {"loss": loss, "accuracy": accuracy}

    def training_epoch_end(self, outputs: List[Dict[str, torch.tensor]]):
        get_logger(self.hparams.experiment_name).info(
            f"========== Training Epoch {self.current_epoch} =========="
        )
        result = self.epoch_end(outputs)
        get_logger(self.hparams.experiment_name).info(str(result))
        return result

    def validation_epoch_end(self, outputs: List[Dict[str, torch.tensor]]):
        get_logger(self.hparams.experiment_name).info(
            f"========== Validation Epoch {self.current_epoch} =========="
        )
        result = self.epoch_end(outputs)
        get_logger(self.hparams.experiment_name).info(str(result))
        return result

    def test_epoch_end(self, outputs: List[Dict[str, torch.tensor]]):
        get_logger(self.hparams.experiment_name).info("========== Test ==========")
        result = self.epoch_end(outputs)
        get_logger(self.hparams.experiment_name).info(str(result))
        return result

    def train_dataloader(self) -> DataLoader:
        ds_train = MedNliTsvDataset(Path(self.hparams.data_dir) / "mli_train_v1.tsv")
        return DataLoader(ds_train, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        ds_val = MedNliTsvDataset(Path(self.hparams.data_dir) / "mli_dev_v1.tsv")
        return DataLoader(ds_val, batch_size=self.hparams.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        ds_test = MedNliTsvDataset(Path(self.hparams.data_dir) / "mli_test_v1.tsv")
        return DataLoader(ds_test, batch_size=self.hparams.batch_size, shuffle=False)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-path", type=str, dest="bert_path", default="")
    parser.add_argument("--lr", type=float, dest="lr", default=5e-5)
    parser.add_argument("--max-epoch", type=int, dest="max_epoch", default=4)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=16)
    parser.add_argument("--cuda", type=int, dest="cuda", default=0)
    parser.add_argument(
        "--experiment-name", type=str, dest="experiment_name", default=""
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
