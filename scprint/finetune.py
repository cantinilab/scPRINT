from lightning.pytorch.callbacks.finetuning import BaseFinetuning
import torch.nn as nn
from torch.optim import Optimizer
import torch.optim as optim


class FineTuneMode(BaseFinetuning):
    def __init__(self, lr: float = 1e-5, optimizer: str = "adamW"):
        super().__init__()
        self.lr = lr
        if optimizer == "adamW":
            self.optimizer = optim.AdamW
        elif optimizer == "adam":
            self.optimizer = optim.Adam
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        # Freeze all parameters in the model
        self.freeze(pl_module, train_bn=False)

    def finetune_function(self, pl_module: "pl.LightningModule") -> None:
        # Get all modules in a flat list
        modules = list(pl_module.modules())
        # Get only the last 2 layers that are nn.Module instances
        last_layers = [
            m
            for m in modules
            if isinstance(m, nn.Module) and len(list(m.parameters(recurse=False))) > 0
        ][-2:]

        # Unfreeze the last 2 layers and add them to optimizer with lower learning rate
        self.unfreeze_and_add_param_group(
            modules=last_layers, optimizer=self.optimizer, lr=self.lr, train_bn=False
        )
