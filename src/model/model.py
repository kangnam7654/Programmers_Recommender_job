from turtle import forward
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

class BinaryClassification(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.build_model()
        self.criterion = nn.BCEWithLogitsLoss()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(421, 600),
            nn.GELU(),
            nn.Linear(600, 1)
        )
        return model
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        return optimizer