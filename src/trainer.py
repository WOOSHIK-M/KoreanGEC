from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.dataset import MyDataset
from src.model import TorchLightningModel


class Trainer:
    """A trainer class."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize."""
        self.config = config

        self.dataset = MyDataset(val_ratio=self.config["val_ratio"])  # type: ignore
        self.model = TorchLightningModel(
            model_name=self.config["model"],
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

    def run(self) -> None:
        """Do train or evaluation."""
        # prepare dataloader
        train_dataset, val_dataset = self.dataset.load(self.model.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], num_workers=4)

        # create trainer and run it!
        trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
        )
        trainer.fit(self.model, train_loader, val_loader)
