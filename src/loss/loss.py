import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles).view(1, -1))
        
    def forward(self, preds, target):
        """
        preds: (Batch, Num_Quantiles)
        target: (Batch) or (Batch, 1)
        """
        target = target.view(-1, 1)
        
        # Calculate errors
        errors = target - preds
        
        # Pinball Loss Logic
        # q * error if underpredicting (error > 0)
        # (q-1) * error if overpredicting (error < 0)
        loss = torch.max(
            (self.quantiles - 1) * errors, 
            self.quantiles * errors
        )
        
        return loss.sum(dim=1).mean()