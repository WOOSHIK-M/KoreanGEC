from abc import abstractmethod
from typing import Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import login as huggingface_login
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import HUGGINGFACE_KEY

huggingface_login(HUGGINGFACE_KEY)


class CustomHuggingfaceModel(nn.Module):
    """An abstract class of custom model."""

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load its tokenizer."""
        pass

    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        """Load its tokenizer."""
        pass

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()


class BertBaseUncased(CustomHuggingfaceModel):
    """A pre-trained model from huggingface.

    https://huggingface.co/google-bert/bert-base-uncased
    """

    # MODEL_ID = "bert-base-uncased"
    MODEL_ID = "bert-base-multilingual-cased"

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward."""
        return self.model(**inputs).logits

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Get a tokenizer of this model."""
        return AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)

    def load_model(self) -> PreTrainedModel:
        """Get a model to be fine-tuned."""
        return AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID, num_labels=2)


class TorchLightningModel(pl.LightningModule):
    """A pytorch-lightning model class."""

    MODEL_DICT: dict[str, Type[CustomHuggingfaceModel]] = {"bertbase": BertBaseUncased}

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get tokenizer of model class."""
        return self.model.tokenizer

    def __init__(
        self,
        model_name: str,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.save_hyperparameters()

        assert model_name in self.MODEL_DICT, f"Undefined model name: {model_name}"

        self.model = self.MODEL_DICT[model_name]()
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Do inference.

        The model returns the probability for each class.
        """
        return self.model(inputs)  # type: ignore

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Do inference for training."""
        inputs, labels = batch
        pred = self(inputs)
        loss = self.loss_fn(pred, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        """Do inference for validation."""
        inputs, labels = batch
        pred = self(inputs)
        loss = self.loss_fn(pred, labels)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> Optimizer:
        """Get optimizer."""
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # type: ignore
