import torch 
from torch import nn
from turtle import forward
from torch import ones_like, zeros_like
from torch.nn import Module, Sigmoid
import torch.nn.functional as F
from torch.nn import MarginRankingLoss, SoftMarginLoss, BCELoss

_loss_name = ['MarginLoss', 'selfAdversarialNegativeSamplingLoss','LogisticLoss','SigmoidLoss']

class MarginLoss(Module):
    def __init__(self, margin):
        super().__init__()
        self.loss = MarginRankingLoss(margin = margin, reduction = 'sum')
    
    def forward(self, positive_triplets, negative_triplets):
        return self.loss(positive_triplets, negative_triplets,
                        target = ones_like(positive_triplets))


class LogisticLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = SoftMarginLoss(reduction='sum')

    def forward(self, positive_score, negative_score):
        targets = ones_like(positive_score)
        return self.loss(positive_score, targets) + \
               self.loss(negative_score, -targets)

class selfAdversarialNegativeSamplingLoss(Module):

    def __init__(self, margin, adv_temperature):
        super().__init__()
        self.margin = margin
        self.adv_temperature = adv_temperature

    def forward(self, positive_score, negative_score):
        positive_item = F.logsigmoid(positive_score)
        negative_item = (F.softmax(negative_score * self.adv_temperature).detach() *
                         F.logsigmoid(- negative_score))

        positive_item_loss = -positive_item.mean()
        negative_item_loss = -negative_item.mean()

        return (positive_item_loss + negative_item_loss)/2
        
                           
class SigmoidLoss(Module):

	def __init__(self, adv_temperature = None):
		super(SigmoidLoss, self).__init__()
		self.criterion = nn.LogSigmoid()
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False

	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2
		else:
			return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2

	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()