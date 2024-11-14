import shutil
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import logging

from src.dataset import CustomDataset, MyDataset
from src.model import TorchLightningModel

logging.set_verbosity_error()
torch.set_float32_matmul_precision("high")


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> None:
        """Do train or evaluation."""
        self.eval() if self.config["eval"] else self.train()

    def train(self) -> None:
        """Fine-tune the model."""
        # prepare dataloader
        train_dataset, val_dataset = self.dataset.load(self.model.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=10,
        )
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), num_workers=10)

        # create trainer and run it!
        save_dir = "save"
        if Path(save_dir).exists():
            shutil.rmtree(save_dir)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(  # type: ignore
            monitor="acc",
            dirpath=save_dir,
            filename="{acc:.4f}",
            save_top_k=3,
            mode="max",
        )
        trainer = pl.Trainer(
            default_root_dir=save_dir,
            max_epochs=self.config["max_epochs"],
            val_check_interval=0.05,
            callbacks=[checkpoint_callback],
            logger=True,
        )
        trainer.fit(self.model, train_loader, val_loader)

    def eval(self) -> None:
        """."""
        trainer = pl.Trainer()

        model = self.model
        if self.config["checkpoint"]:
            model = TorchLightningModel.load_from_checkpoint(self.config["checkpoint"])
        model.to(self.device)

        if self.config["test"]:
            self.do_inference_with_txt(self.config["test"], model)
        else:
            val_dataset = self.dataset.load(self.model.tokenizer)[1]
            val_loader = DataLoader(
                val_dataset, batch_size=self.config["batch_size"], num_workers=4
            )
            trainer.validate(model, val_loader)

    def do_inference_with_txt(self, fpath, model):
        with open(fpath) as f:
            for sentence in f:
                sentence = sentence.strip()[:-1]

                inputs = CustomDataset.tokenize(sentence, model.tokenizer)
                inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
                outputs = model(inputs)

                answer = outputs.squeeze().argmax().bool().item()
                print(f"[Text] {sentence} ({'O' if answer else 'X'})")
