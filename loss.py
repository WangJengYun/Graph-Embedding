from torch import ones_like, zeros_like
from torch.nn import Module, Sigmoid
from torch.nn import MarginRankingLoss, SoftMarginLoss, BCELoss

class MarginLoss(Module):
    def __init__(self, margin):
        super().__init__()
        self.loss = MarginRankingLoss(margin = margin, reduction = 'sum')
    
    def forward(self, positive_triplets, negative_triplets):
        return self.loss(positive_triplets, negative_triplets,
                        target = ones_like(positive_triplets))
