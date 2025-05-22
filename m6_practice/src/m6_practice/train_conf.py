import matplotlib.pyplot as plt
import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel

import hydra
import os

from omegaconf import OmegaConf
# loading
config = OmegaConf.load('conf/config.yaml')

import logging
logging.basicConfig(filename='mnist.log', level=logging.INFO,)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
batch_size=config.hyperparameters.batch_size  
epochs=config.hyperparameters.epochs
save_path=config.hyperparameters.save_path

@hydra.main(config_name="adam.yaml", config_path=f"{os.getcwd()}/conf/optimizer", version_base="1.3")
def train(cfg) -> None:
    """Train a model on MNIST."""
    log.info("Training day and night")
    # log.info(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    log.info("Training complete")
    torch.save(model.state_dict(), save_path)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()