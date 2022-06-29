import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

class BinaryClassification(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.build_model()
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(421, 1028),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1028, 2056),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(2056, 4112),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(4112, 2)
        )
        return model
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        cls = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        acc = (cls.unsqueeze(1) == y).sum() / cls.shape[0]
        y = y.squeeze(1).long()
        loss = self.criterion(logits, y)
        
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        cls = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        acc = (cls.unsqueeze(1) == y).sum() / cls.shape[0]
        y = y.squeeze(1).long()
        loss = self.criterion(logits, y)
        self.log('valid_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        pass
    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        cls = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        return cls
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.0001)
        return optimizer