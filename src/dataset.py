import json
from abc import abstractmethod
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from transformers import PreTrainedTokenizer

DATA_DIR = "data"


class CustomDataset(Dataset):
    """A abstract class of custom dataset."""

    @abstractmethod
    def count_data(self) -> int:
        """Count the number of data."""
        pass

    @abstractmethod
    def get_item(self, idx: int) -> tuple[str, int]:
        """Get a data of the given index.

        It should return a sentence and its correctness.
        """
        pass

    def __init__(self, data: list | dict | pd.DataFrame, tokenizer: PreTrainedTokenizer) -> None:
        """Initializer."""
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Get the number of data of this."""
        return self.count_data()

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Get a data of the given index."""
        sentence, label = self.get_item(idx)
        inputs = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        inputs = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
        label = torch.tensor(label)
        return inputs, label


class KNCTDataset(CustomDataset):
    """A k-nct dataset."""

    FILE_PATH = "K-NCT_v1.4.json"

    @staticmethod
    def load(tokenizer: PreTrainedTokenizer) -> "KNCTDataset":
        """Load this dataset."""
        file_path = Path(DATA_DIR) / KNCTDataset.FILE_PATH
        assert file_path.exists, f"Not found, {file_path}"

        with open(file_path) as f:
            raw_data = json.load(f)["data"]
        return KNCTDataset(raw_data, tokenizer)

    def count_data(self) -> int:
        """Count the number of data.

        Each data has 2 sentences, correct and incorrect.
        """
        return len(self.data) * 2

    def get_item(self, idx: int) -> tuple[str, int]:
        """Get a item."""
        idx, is_correct = divmod(idx, 2)

        info = self.data[idx]
        sentence = info["correct_sentence"] if is_correct else info["error_sentence"]
        return sentence, is_correct


class MyDataset:
    """A dataset to be used to fine-tuning."""

    def __init__(self, val_ratio: float = 0.3) -> None:
        """Initialize."""
        self.val_ratio = val_ratio

    def load(self, tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset]:
        """Load all train and val dataset."""
        knct_dataset = KNCTDataset.load(tokenizer)
        dataset = knct_dataset

        # split train & val dataset
        n_val = int(len(dataset) * self.val_ratio)
        train_dataset = Subset(dataset, range(0, len(dataset) - n_val))
        val_dataset = Subset(dataset, range(len(dataset) - n_val, len(dataset)))
        return train_dataset, val_dataset