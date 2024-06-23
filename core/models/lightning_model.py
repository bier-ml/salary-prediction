from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from core.embedding_models import DummyEmbeddingModel
from core.models.base_model import BaseMatchingModel
from core.models.tools import preprocess_dataframe


class PlDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.dataframe.iloc[idx, 0]),
            torch.FloatTensor([self.dataframe.iloc[idx, 1]]),
        )


class SimilarityNet(pl.LightningModule):
    def __init__(self, input_dim: int = 624):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, 256)
        self.hidden_layer = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return torch.sigmoid(self.output_layer(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        logs = {"val_loss": loss, "mse": mse}
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return logs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


class TorchMatchingModel(BaseMatchingModel):
    def __init__(self, embedding_model=None):
        self.model = None
        self.embedding_model = embedding_model

    def predict(self, vacancy: str | np.ndarray) -> float:
        if self.embedding_model is None:
            self.embedding_model = DummyEmbeddingModel()

        if isinstance(vacancy, str):
            vacancy = self.embedding_model.generate(vacancy)

        emb = vacancy
        emb = torch.from_numpy(emb).float().unsqueeze(0)

        with torch.no_grad():
            return self.model(emb)[0].numpy()

    def load_model(self, model_class: pl.LightningModule, checkpoint_path: Path | str):
        self.model = model_class.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.model.eval()
        print("Model successfully loaded")


if __name__ == "__main__":
    data_path = Path("../../data/dataset_processed.csv")
    df = pd.read_csv(data_path)[["embedding", "similarity"]]
    df = preprocess_dataframe(df)

    train_df, valid_df = train_test_split(df, test_size=0.2)

    train_dataset = PlDataset(train_df)
    valid_dataset = PlDataset(valid_df)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model = SimilarityNet().to(device)
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_dataloader)

    trainer.save_checkpoint("../../data/model_weights/SimilarityNet.ckpt")
