import os

import click

from src.trainer import Trainer


@click.group()
def main():
    """Run group of CLI."""


@click.command()
@click.option("--eval", is_flag=True, default=False, help="Evaluation mode.")
@click.option("--checkpoint", type=str, default="", help="Checkpoint path.")
@click.option("--test", type=str, default="", help="test file path.")
@click.option("--model", type=str, default="bertbase", help="Choose a model to be fine-tuned.")
@click.option("--max-epochs", type=int, default=3, help="The maximum epochs to fine-tune.")
@click.option("--val-ratio", type=float, default=0.3, help="The ratio of validate dataset.")
@click.option("--batch-size", type=int, default=8, help="Batch size.")
@click.option("--lr", type=float, default=1e-5, help="Learning rate.")
@click.option("--weight-decay", type=float, default=0.01, help="Weight decay of learning rate.")
@click.option("--gpu", type=str, default="", help="Choose gpu(s) to be used(e.g. --gpu 0,1).")
def run(**config) -> None:
    """Run."""
    if config["gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

    Trainer(config).run()


if __name__ == "__main__":
    main.add_command(run)

    main()
