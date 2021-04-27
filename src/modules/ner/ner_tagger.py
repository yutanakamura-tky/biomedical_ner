#!/usr/bin/env python
# coding: utf-8

# How to use this script
#
# 1. Place dataset in the following directories and file names:
#   - xxx.text
#       - raw text file.
#   - xxx.tokens
#       - text file containing tokenized file with each tokens separated with spaces.
#       - it is useful to
#       - e.g., John Smith is a 76-year old male .
#   - xxx.ann
#       - text file containing gold BIO tagging separated with commas.
#       - e.g., O,O,O,O,B,I,O,O,O,O,O
#
#

# # 0. Preparation

# ### 0-1. Dependencies

import glob
import itertools
import pathlib
import re
import shlex
import subprocess
from logging import DEBUG, INFO, FileHandler, Formatter, StreamHandler, getLogger
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
from allennlp.modules import conditional_random_field
from allennlp.modules.elmo import Elmo, batch_to_ids
from seqeval.metrics import classification_report as seq_classification_report
from sklearn.metrics import classification_report
from transformers import BertConfig, BertForPreTraining, BertTokenizer

# ### 0-2. Prepare for logging


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


# ### 0-3. NER tag
# 'BOS' and 'EOS' are not needed to be explicitly included

ID_TO_LABEL = {0: "O", 1: "I-MEDICATION", 2: "B-MEDICATION"}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


def tag_to_id(tag, ltoi):
    """
    str, dict -> torch.tensor
    input:
        tag (str): BIO tagging (e.g., 'O,O,B,I,I,O,...')
        itol (dict): LABEL-TO-ID Mapping
    output:
        torch.Tensor: IOB2 tagging (e.g., ([0,0,2,1,1,0,...]))
    """

    # sequence of 0 (O-tag) or 2 (B-P-tag) or 1 (I-P-tag)
    # e.g., '2,1,0,2,1,0,...'
    tag = re.sub("B", f"{ltoi['B-MEDICATION']}", tag)
    tag = re.sub("I", f"{ltoi['I-MEDICATION']}", tag)
    tag = re.sub("O", f"{ltoi['O']}", tag)

    return torch.tensor([int(t) for t in tag.split(",")])


# ### 1. Dataset, DataLoader


class NERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        text_file_paths: List[str],
        token_file_paths: List[str],
        tag_file_paths: List[str],
    ):
        """
        text_file_paths: list(str)
        token_file_paths: list(str)
        tag_file_paths: list(str)
        """
        self.text_file_paths = text_file_paths
        self.token_file_paths = token_file_paths
        self.tag_file_paths = tag_file_paths
        self.ix = [
            int(re.compile(r"/([0-9]+)[^0-9]+").findall(path)[0])
            for path in self.text_file_paths
        ]
        assert (
            len(text_file_paths) == len(token_file_paths) == len(tag_file_paths)
        ), "ERROR: All arguments must be lists of the same size."
        assert len(text_file_paths) > 0, "ERROR: Passed file lists are empty."
        self.n = len(text_file_paths)

        self.itol = ID_TO_LABEL
        self.ltoi = {v: k for k, v in self.itol.items()}

    @classmethod
    def from_dirnames(cls, dirnames: List[str]):
        """
        requirements:
            Data must be in the format below:
                - foo.text: raw text of an XML record without NER tagging.
                - foo.tokens: a tokenized XML record without NER tagging.
                - foo.ann: IOB tagging of the record (e.g. O,O,O,B,I,O,O,...)
        """
        text_paths = sorted(
            list(
                itertools.chain(
                    *[glob.glob(str(pathlib.Path(d) / "*.text")) for d in dirnames]
                )
            )
        )
        token_paths = sorted(
            list(
                itertools.chain(
                    *[glob.glob(str(pathlib.Path(d) / "*.tokens")) for d in dirnames]
                )
            )
        )
        tag_paths = sorted(
            list(
                itertools.chain(
                    *[glob.glob(str(pathlib.Path(d) / "*.ann")) for d in dirnames]
                )
            )
        )
        paths = (text_paths, token_paths, tag_paths)
        return cls(*paths)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        returns = {}
        returns["ix"] = self.ix[idx]

        with open(self.text_file_paths[idx]) as f:
            # Raw document. Example:
            # [Triple therapy regimens involving H2 blockaders for therapy of Helicobacter pylori infections].
            # Comparison of ranitidine and lansoprazole in
            # short-term low-dose triple therapy for Helicobacter pylori infection.
            returns["text"] = f.read()

        with open(self.token_file_paths[idx]) as f:
            # Tokenized document. Example:
            # ['[', 'Triple', 'therapy', 'regimens', 'involving', 'H2', 'blockaders',
            #  'for', 'therapy', 'of', 'Helicobacter', 'pylori', 'infections', ']', '.',
            #  'Comparison', 'of', 'ranitidine', 'and', 'lansoprazole', 'in',
            #  'short-term', 'low-dose', 'triple', 'therapy', 'for',
            #  'Helicobacter', 'pylori', 'infection', '.', ...
            tokens = f.read().split()
            returns["tokens"] = tokens

        with open(self.tag_file_paths[idx]) as f:
            # sequence of B,I,O tags
            # e.g., 'O,O,B,I,O,...'
            tag = f.read()

        # torch.tensor of IBO2 tag (e.g., ([0,0,2,1,0,...]))
        tag = tag_to_id(tag, self.ltoi)
        returns["tags"] = tag.numpy().tolist()

        return returns


class NERDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        """
        text_file_paths: list(str)
        token_file_paths: list(str)
        p_files: list(str)
        i_files: list(str)
        o_files: list(str)
        """
        kwargs["collate_fn"] = lambda batch: {
            "ix": torch.tensor([sample["ix"] for sample in batch]),
            "text": [sample["text"] for sample in batch],
            "tokens": [sample["tokens"] for sample in batch],
            "tags": [sample["tags"] for sample in batch],
        }
        super().__init__(dataset, **kwargs)


# 2. LightningModule


class NERTagger(pl.LightningModule):
    def __init__(self, hparams):
        """
        input:
            hparams: namespace with the following items:
                'data_dir' (str): Data Directory. default: './official/ebm_nlp_1_00'
                'bioelmo_dir' (str): BioELMo Directory. default: './models/bioelmo', help='BioELMo Directory')
                'max_length' (int): Max Length. default: 1024
                'lr' (float): Learning Rate. default: 1e-2
                'fine_tune_bioelmo' (bool): Whether to Fine Tune BioELMo. default: False
                'lr_bioelmo' (float): Learning Rate in BioELMo Fine-tuning. default: 1e-4
        """
        super().__init__()
        self.hparams = hparams
        self.itol = ID_TO_LABEL
        self.ltoi = {v: k for k, v in self.itol.items()}

        if self.hparams.model == "bioelmo":
            # Load Pretrained BioELMo
            DIR_ELMo = pathlib.Path(str(self.hparams.bioelmo_dir))
            self.bioelmo = self.load_bioelmo(
                DIR_ELMo, not self.hparams.fine_tune_bioelmo
            )
            self.bioelmo_output_dim = self.bioelmo.get_output_dim()

            # ELMo Padding token (In ELMo token with ID 0 is used for padding)
            VOCAB_FILE_PATH = DIR_ELMo / "vocab.txt"
            command = shlex.split(f"head -n 1 {VOCAB_FILE_PATH}")
            res = subprocess.Popen(command, stdout=subprocess.PIPE)
            self.bioelmo_pad_token = res.communicate()[0].decode("utf-8").strip()

            # Initialize Intermediate Affine Layer
            self.hidden_to_tag = nn.Linear(int(self.bioelmo_output_dim), len(self.itol))

        elif self.hparams.model == "biobert":
            # Load Pretrained BioBERT
            PATH_BioBERT = pathlib.Path(str(self.hparams.biobert_path))
            self.bertconfig = BertConfig.from_pretrained(self.hparams.bert_model_type)
            self.bertforpretraining = BertForPreTraining(self.bertconfig)
            self.bertforpretraining.load_tf_weights(self.bertconfig, PATH_BioBERT)
            self.biobert = self.bertforpretraining.bert
            self.tokenizer = BertTokenizer.from_pretrained(self.hparams.bert_model_type)

            # Freeze BioBERT if fine-tune not desired
            if not self.hparams.fine_tune_biobert:
                for n, m in self.biobert.named_parameters():
                    m.requires_grad = False

            # Initialize Intermediate Affine Layer
            self.hidden_to_tag = nn.Linear(
                int(self.bertconfig.hidden_size), len(self.itol)
            )

        # Initialize CRF
        TRANSITIONS = conditional_random_field.allowed_transitions(
            constraint_type="BIO", labels=self.itol
        )
        self.crf = conditional_random_field.ConditionalRandomField(
            # set to 3 because here "tags" means ['O', 'B', 'I']
            # no need to include 'BOS' and 'EOS' in "tags"
            num_tags=len(self.itol),
            constraints=TRANSITIONS,
            include_start_end_transitions=False,
        )
        self.crf.reset_parameters()

    @staticmethod
    def load_bioelmo(bioelmo_dir: str, freeze: bool) -> Elmo:
        # Load Pretrained BioELMo
        DIR_ELMo = pathlib.Path(bioelmo_dir)
        bioelmo = Elmo(
            DIR_ELMo / "biomed_elmo_options.json",
            DIR_ELMo / "biomed_elmo_weights.hdf5",
            1,
            requires_grad=bool(not freeze),
            dropout=0,
        )
        return bioelmo

    def get_device(self):
        return self.crf.state_dict()["transitions"].device

    def _forward_bioelmo(self, tokens) -> Tuple[torch.Tensor, torch.Tensor]:
        # character_ids: torch.tensor(n_batch, len_max)
        # documents will be padded to have the same token lengths as the longest document
        character_ids = batch_to_ids(tokens)
        character_ids = character_ids[:, : self.hparams.max_length, :]
        character_ids = character_ids.to(self.get_device())

        # characted_ids -> BioELMo hidden state of the last layer & mask
        out = self.bioelmo(character_ids)
        hidden = out["elmo_representations"][-1]
        crf_mask = out["mask"].to(torch.bool).to(self.get_device())

        return (hidden, crf_mask)

    def _forward_biobert(
        self, tokens: List[List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return BioBERT Hidden state for the tokenized documents.
        Documents with different lengths will be accepted.

        list(list(str)) -> tuple(torch.tensor, torch.tensor)
        """
        # Convert each token of each document into a list of subwords.
        # e.g.,
        #   [['Admission', 'Date', ...], ['Service', ':', ...]]
        #       |
        #       V
        #   [[['Ad', '##mission'], ['Date'], ...], [['Service'], [':'], ...]]
        subwords_unchained = [
            [self.tokenizer.tokenize(tok) for tok in doc] for doc in tokens
        ]

        # Simply replace each token of each document with corresponding subwords.
        # e.g.,
        #   [['Admission', 'Date', ...], ['Service', ':', ...]]
        #       |
        #       V
        #   [['Ad', '##mission', 'Date', ...], ['Service', ':', ...]]
        subwords = [
            list(itertools.chain(*[self.tokenizer.tokenize(tok) for tok in doc]))
            for doc in tokens
        ]

        # Memorize (i) header place of each token and (ii) how many subwords each token gave birth.
        # e.g.,
        #   For document ['Admission', 'Date'] -> ['Ad', '##mission', 'Date'],
        #   subword_info will be {'start':[0,2], 'length':[2,1]}.
        subword_info = []
        for doc in subwords_unchained:
            word_lengths = [len(word) for word in doc]
            word_head_ix = [0]
            for i in range(len(word_lengths) - 1):
                word_head_ix.append(word_head_ix[-1] + word_lengths[i])
            assert len(word_lengths) == len(word_head_ix)
            subword_info.append({"start": word_head_ix, "length": word_lengths})

        assert [len(info["start"]) for info in subword_info] == [
            len(doc) for doc in tokens
        ]

        # Split each document into chunks shorter than max_length.
        # Here, each document will be simply split at every 510 tokens.

        max_length = min(
            self.bertconfig.max_position_embeddings, self.hparams.max_length
        )

        longest_length = max([len(doc) for doc in subwords])
        n_chunks = (longest_length - 1) // (max_length - 2) + 1
        chunks = []
        for n in range(n_chunks):
            chunk_of_all_documents = []
            for document in subwords:
                chunk_of_single_document = document[
                    (max_length - 2) * n : (max_length - 2) * (n + 1)
                ]
                if chunk_of_single_document == []:
                    chunk_of_all_documents.append([""])
                else:
                    chunk_of_all_documents.append(chunk_of_single_document)
            chunks.append(chunk_of_all_documents)

        # Convert chunks into BERT input form.
        inputs = []
        for chunk in chunks:
            if type(chunk) is str:
                unsqueezed_chunk = [[chunk]]
            elif type(chunk) is list:
                if type(chunk[0]) is str:
                    unsqueezed_chunk = [chunk]
                elif type(chunk[0]) is list:
                    unsqueezed_chunk = chunk

            inputs.append(
                self.tokenizer.batch_encode_plus(
                    unsqueezed_chunk,
                    pad_to_max_length=True,
                    is_pretokenized=True,
                )
            )

        # Get BioBERT hidden states.
        hidden_states = []
        for inpt in inputs:
            inpt_tensors = {
                k: torch.tensor(v).to(self.get_device()) for k, v in inpt.items()
            }
            hidden_state = self.biobert(**inpt_tensors)[0][:, 1:-1, :]
            hidden_states.append(hidden_state)

        # Concatenate hidden states from each chunk.
        hidden_states_cat = torch.cat(hidden_states, dim=1)

        # If a word was tokenized into multiple subwords, take average of them.
        # e.g. Hidden state for "Admission" equals average of hidden states for "Ad" and "##mission"
        hidden_states_shrunk = torch.zeros_like(hidden_states_cat)
        for n in range(hidden_states_cat.size()[0]):
            hidden_state_shrunk = torch.stack(
                [
                    torch.narrow(hidden_states_cat[n], dim=0, start=s, length=l).mean(
                        dim=0
                    )
                    for s, l in zip(subword_info[n]["start"], subword_info[n]["length"])
                ]
            )
            hidden_states_shrunk[
                n, : hidden_state_shrunk.size()[0], :
            ] = hidden_state_shrunk

        # Truncate lengthy tail that will not be used.
        hidden_states_shrunk = hidden_states_shrunk[
            :, : max([len(doc) for doc in tokens]), :
        ]

        # Create mask for CRF.
        crf_mask = torch.zeros(hidden_states_shrunk.size()[:2]).to(torch.uint8)
        for i, length in enumerate([len(doc) for doc in tokens]):
            crf_mask[i, :length] = 1
        crf_mask = crf_mask > 0
        crf_mask = crf_mask.to(self.get_device())

        return (hidden_states_shrunk, crf_mask)

    def _forward_crf(
        self,
        hidden: torch.Tensor,
        gold_tags_padded: torch.Tensor,
        crf_mask: torch.Tensor,
    ) -> Dict:
        """
        input:
            hidden (torch.tensor) (n_batch, seq_length, hidden_dim)
            gold_tags_padded (torch.tensor) (n_batch, seq_length)
            crf_mask (torch.bool) (n_batch, seq_length)
        output:
            result (dict)
                'log_likelihood' : torch.tensor
                'pred_tags_packed' : torch.nn.utils.rnn.PackedSequence
                'gold_tags_padded' : torch.tensor
        """
        result = {}

        if not (hidden.size()[1] == gold_tags_padded.size()[1] == crf_mask.size()[1]):
            raise RuntimeError(
                "seq_length of hidden, gold_tags_padded, and crf_mask do not match: "
                + f"{hidden.size()}, {gold_tags_padded.size()}, {crf_mask.size()}"
            )

        if gold_tags_padded is not None:
            # Training Mode
            # Log likelihood
            log_prob = self.crf.forward(hidden, gold_tags_padded, crf_mask)

            # top k=1 tagging
            Y = [
                torch.tensor(result[0])
                for result in self.crf.viterbi_tags(logits=hidden, mask=crf_mask)
            ]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)

            result["log_likelihood"] = log_prob
            result["pred_tags_packed"] = Y
            result["gold_tags_padded"] = gold_tags_padded
            return result

        else:
            # Prediction Mode
            # top k=1 tagging
            Y = [
                torch.tensor(result[0])
                for result in self.crf.viterbi_tags(logits=hidden, mask=crf_mask)
            ]
            Y = rnn.pack_sequence(Y, enforce_sorted=False)
            result["pred_tags_packed"] = Y
            return result

    def forward(self, tokens, gold_tags=None):
        """
        Main NER tagging function.
        Documents with different token lengths are accepted.

        input:
            tokens (list(list(str))): List of documents for the batch. Each document must be stored as a list of tokens.
            gold_tags (list(list(int))): List of gold labels for each document of the batch.
        output:
            result (dict)
                'log_likelihood' : torch.tensor
                'pred_tags_packed' : torch.nn.utils.rnn.PackedSequence
                'gold_tags_padded' : torch.tensor
        """
        if self.hparams.model == "bioelmo":
            # BioELMo features
            hidden, crf_mask = self._forward_bioelmo(tokens)

        elif self.hparams.model == "biobert":
            # BioELMo features
            hidden, crf_mask = self._forward_biobert(tokens)

        # Turn on gradient tracking
        # Affine transformation (Hidden_dim -> N_tag)
        hidden.requires_grad_()
        hidden = self.hidden_to_tag(hidden)

        if gold_tags is not None:
            gold_tags = [torch.tensor(seq) for seq in gold_tags]
            gold_tags_padded = rnn.pad_sequence(
                gold_tags, batch_first=True, padding_value=self.ltoi["O"]
            )
            gold_tags_padded = gold_tags_padded[:, : self.hparams.max_length]
            gold_tags_padded = gold_tags_padded.to(self.get_device())
        else:
            gold_tags_padded = None

        result = self._forward_crf(hidden, gold_tags_padded, crf_mask)
        return result

    def recognize_named_entity(self, token, gold_tags=None):
        """
        Alias of self.forward().
        """
        return self.forward(token, gold_tags)

    def step(self, batch, batch_nb, *optimizer_idx):
        tokens_nopad = batch["tokens"]
        tags_nopad = batch["tags"]

        assert list(map(len, tokens_nopad)) == list(
            map(len, tags_nopad)
        ), "ERROR: the number of tokens and BIO tags are different in some record."

        # Negative Log Likelihood
        result = self.forward(tokens_nopad, tags_nopad)
        returns = {
            "loss": result["log_likelihood"] * (-1.0),
            "T": result["gold_tags_padded"],
            "Y": result["pred_tags_packed"],
            "I": batch["ix"],
        }

        assert (
            torch.isnan(returns["loss"]).sum().item() == 0
        ), "Loss function contains nan."
        return returns

    def unpack_pred_tags(self, Y_packed):
        """
        input:
            Y_packed: torch.nn.utils.rnn.PackedSequence
        output:
            Y: list(list(str))
                Predicted NER tagging sequence.
        """
        Y_padded, Y_len = rnn.pad_packed_sequence(
            Y_packed, batch_first=True, padding_value=-1
        )
        Y_padded = Y_padded.numpy().tolist()
        Y_len = Y_len.numpy().tolist()

        # Replace B- tag with I- tag
        # because the original paper defines the NER task as identification of spans, not entities
        Y = [
            [self.itol[ix].replace("B-", "I-") for ix in ids[:length]]
            for ids, length in zip(Y_padded, Y_len)
        ]

        return Y

    def unpack_gold_and_pred_tags(self, T_padded, Y_packed):
        """
        input:
            T_padded: torch.tensor
            Y_packed: torch.nn.utils.rnn.PackedSequence
        output:
            T: list(list(str))
                Gold NER tagging sequence.
            Y: list(list(str))
                Predicted NER tagging sequence.
        """
        Y = self.unpack_pred_tags(Y_packed)
        Y_len = [len(seq) for seq in Y]

        T_padded = T_padded.numpy().tolist()

        # Replace B- tag with I- tag
        # because the original paper defines the NER task as identification of spans, not entities
        T = [
            [self.itol[ix] for ix in ids[:length]]
            for ids, length in zip(T_padded, Y_len)
        ]

        return T, Y

    def gather_outputs(self, outputs):
        if len(outputs) > 1:
            loss = torch.mean(torch.tensor([output["loss"] for output in outputs]))
        else:
            loss = outputs[0]["loss"]

        IX = []
        Y = []
        T = []

        for output in outputs:
            T_batch, Y_batch = self.unpack_gold_and_pred_tags(
                output["T"].cpu(), output["Y"].cpu()
            )
            T += T_batch
            Y += Y_batch
            IX += output["I"].cpu().numpy().tolist()

        returns = {"loss": loss, "T": T, "Y": Y, "I": IX}

        return returns

    def training_step(self, batch, batch_nb, *optimizer_idx) -> Dict:
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """
        return self.step(batch, batch_nb, *optimizer_idx)

    def training_epoch_end(self, outputs: Union[List[Dict], List[List[Dict]]]) -> Dict:
        """
        outputs(list of dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        outs = self.gather_outputs(outputs)
        loss = outs["loss"]
        Y = outs["Y"]
        T = outs["T"]

        get_logger(self.hparams.version).info(
            f"========== Training Epoch {self.current_epoch} =========="
        )
        get_logger(self.hparams.version).info(f"Loss: {loss.item()}")
        get_logger(self.hparams.version).info(
            f"Entity-wise classification report\n{seq_classification_report(T, Y, 4)}"
        )

        progress_bar = {"train_loss": loss}
        returns = {"loss": loss, "progress_bar": progress_bar}
        return returns

    def validation_step(self, batch, batch_nb) -> Dict:
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        """
        return self.step(batch, batch_nb)

    def validation_epoch_end(
        self, outputs: Union[List[Dict], List[List[Dict]]]
    ) -> Dict:
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """
        outs = self.gather_outputs(outputs)
        loss = outs["loss"]
        Y = outs["Y"]
        T = outs["T"]

        get_logger(self.hparams.version).info(
            f"========== Validation Epoch {self.current_epoch} =========="
        )
        get_logger(self.hparams.version).info(f"Loss: {loss.item()}")
        get_logger(self.hparams.version).info(
            f"Entity-wise classification report\n{seq_classification_report(T, Y, 4)}"
        )

        progress_bar = {"val_loss": loss}
        returns = {"val_loss": loss, "progress_bar": progress_bar}
        return returns

    def test_step(self, batch, batch_nb) -> Dict:
        # Process on individual mini-batches
        """
        (batch) -> (dict or OrderedDict)
        """
        return self.step(batch, batch_nb)

    def test_epoch_end(self, outputs: Union[List[Dict], List[List[Dict]]]) -> Dict:
        """
        For single dataloader:
            outputs(list of dict) -> (dict or OrderedDict)
        For multiple dataloaders:
            outputs(list of (list of dict)) -> (dict or OrderedDict)
        """
        outs = self.gather_outputs(outputs)
        loss = outs["loss"]
        Y = outs["Y"]
        T = outs["T"]

        get_logger(self.hparams.version).info("========== Test ==========")
        get_logger(self.hparams.version).info(f"Loss: {loss.item()}")
        get_logger(self.hparams.version).info(
            f"Entity-wise classification report\n{seq_classification_report(T, Y, 4)}"
        )

        progress_bar = {"test_loss": loss}
        returns = {"test_loss": loss, "progress_bar": progress_bar}
        return returns

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]:
        if self.hparams.model == "bioelmo":
            if self.hparams.fine_tune_bioelmo:
                optimizer_bioelmo_1 = optim.Adam(
                    self.bioelmo.parameters(), lr=float(self.hparams.lr_bioelmo)
                )
                optimizer_bioelmo_2 = optim.Adam(
                    self.hidden_to_tag.parameters(), lr=float(self.hparams.lr_bioelmo)
                )
                optimizer_crf = optim.Adam(
                    self.crf.parameters(), lr=float(self.hparams.lr)
                )
                return [optimizer_bioelmo_1, optimizer_bioelmo_2, optimizer_crf]
            else:
                optimizer = optim.Adam(self.parameters(), lr=float(self.hparams.lr))
                return optimizer

        elif self.hparams.model == "biobert":
            if self.hparams.fine_tune_biobert:
                optimizer_biobert_1 = optim.Adam(
                    self.biobert.parameters(), lr=float(self.hparams.lr_biobert)
                )
                optimizer_biobert_2 = optim.Adam(
                    self.hidden_to_tag.parameters(), lr=float(self.hparams.lr_biobert)
                )
                optimizer_crf = optim.Adam(
                    self.crf.parameters(), lr=float(self.hparams.lr)
                )
                return [optimizer_biobert_1, optimizer_biobert_2, optimizer_crf]
            else:
                optimizer = optim.Adam(self.parameters(), lr=float(self.hparams.lr))
                return optimizer

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        ds_train = NERDataset.from_dirnames(self.hparams.train_dirs)
        dl_train = NERDataLoader(
            ds_train, batch_size=self.hparams.batch_size, shuffle=True
        )
        return dl_train

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        ds_val = NERDataset.from_dirnames(self.hparams.val_dirs)
        dl_val = NERDataLoader(
            ds_val, batch_size=self.hparams.batch_size, shuffle=False
        )
        return dl_val

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        ds_test = NERDataset.from_dirnames(self.hparams.test_dirs)
        dl_test = NERDataLoader(
            ds_test, batch_size=self.hparams.batch_size, shuffle=False
        )
        return dl_test


def span_classification_report(T, Y, digits=4):
    """
    Token-wise metrics of NER IOE1 tagging task.
    T: list(list(str)) True labels
    Y: list(list(str)) Pred labels
    """
    T_flatten = []
    Y_flatten = []
    n_sample = len(T)

    for i in range(n_sample):
        T_flatten += T[i]
        Y_flatten += Y[i]

    label_types = [label_type for label_type in set(T_flatten) if label_type != "O"]

    return classification_report(
        T_flatten, Y_flatten, labels=label_types, digits=digits
    )
