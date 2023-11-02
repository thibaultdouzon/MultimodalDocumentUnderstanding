import functools
from math import log
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
import lightning.pytorch as pl
import matplotlib.pyplot as plt

import dall_e
from dall_e import encoder, decoder, utils

BATCH_SIZE = 10
LEARNING_RATE = 7e-3
DEVICE = torch.device("cpu")


def log_fn(message):
    def decorator(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            logger.info(f"BEGIN: {message}")
            try:
                res = fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"FAILED: {message}")
                raise Exception from e
            else:
                logger.success(f"DONE: {message}")
                return res
        return wrap
    return decorator


class PreProcess():
    def __call__(self, x: torch.Tensor):
        x = x.repeat((1, 3, 1, 1))
        x = utils.map_pixels(x)
        x = x.squeeze(0)
        return x


@log_fn("Dataset")
def get_data():

    transform = transforms.Compose([
        transforms.ToTensor(),
        PreProcess(),
        transforms.Resize(128, antialias=True, interpolation=transforms.InterpolationMode.BILINEAR)
    ])
    train_mnist_data = datasets.MNIST(
        "data", download=True, train=True, transform=transform
    )
    test_mnist_data = datasets.MNIST(
        "data", download=True, train=False, transform=transform
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_mnist_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_mnist_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    return train_data_loader, test_data_loader

@log_fn("Model")
def get_model():
    enc = dall_e.load_model("https://cdn.openai.com/dall-e/encoder.pkl", DEVICE)
    dec = dall_e.load_model("https://cdn.openai.com/dall-e/decoder.pkl", DEVICE)
    return enc, dec

class DiscreteVAE(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        enc, dec = get_model()
        self._encoder = enc
        self._decoder = dec

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, _ = batch
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = F.mse_loss(x_hat, x)

        self.log("train_loss", loss.detach())

        return loss
    
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def encode(self, x):
        z_logits = self._encoder(x)
        z = torch.argmax(z_logits, dim=1)
        return z

    def decode(self, z):
        z = F.one_hot(z, num_classes=self._encoder.vocab_size).permute(0, 3, 1, 2).float()
        x_stats = self._decoder(z).float()
        y = utils.unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return y


def display(model, data, fname):
    batch = next(iter(data))
    x, _ = batch

    x_hat = model.decode(model.encode(x.to(model.device)))

    x = x.to("cpu")
    x_hat = x_hat.detach().to("cpu")
    fig, ax = plt.subplots(10, 2)
    for i in range(10):
        ax[i,0].imshow(x[i].permute(1, 2, 0))
        ax[i,1].imshow(x_hat[i].permute(1, 2, 0))
    plt.savefig(fname)

if __name__ == "__main__":
    autoencoder = DiscreteVAE()
    train_dl, test_dl = get_data()
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=1,
        max_steps=10,
        log_every_n_steps=10,
    )
    display(autoencoder, test_dl, "before.png")
    trainer.fit(autoencoder, train_dataloaders=train_dl)
    display(autoencoder, test_dl, "after.png")