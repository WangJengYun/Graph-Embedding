import torch 
from torch import nn
from torch.nn.init import xavier_uniform
from torch.nn.functional import normalize
from scoring_function import _distance_name,l1_distance, l2_distance
from loss_function import _loss_name, MarginLoss, LogisticLoss, SigmoidLoss, selfAdversarialNegativeSamplingLoss

class Base_Model(nn.Module):
    def __init__(self, config, n_entities, n_relations):
        super(Base_Model, self).__init__()
        
        self.config = config
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.distance = self.distance_fn()
        

    def forward(self, positive_triplets, negative_triplets):
        
        heads, tails, relations = positive_triplets
        negative_head, negative_tails, negative_relations = negative_triplets


        positive_score = self.scoring_fn(heads, tails, relations)
        negative_score = self.scoring_fn(negative_head, negative_tails, negative_relations)

        return positive_score, negative_score

    @staticmethod 
    def init_embedding(n_vector, dim):
        
        entity_embeddings = nn.Embedding(n_vector, dim)
        xavier_uniform(entity_embeddings.weight.data)

        return entity_embeddings
      
    def distance_fn(self):
        
        assert hasattr(self.config,'distance_name'), 'Please check param of distance_name'
        if self.config.distance_name in _distance_name:
            if self.config.distance_name == 'L1':
                return l1_distance

            elif self.config.distance_name == 'L2':
                return l2_distance
        else:
            raise ValueError('model cannot instantiate unsupported distance: {}'.format(self.config.distance_name))
    
    def loss_fn(self):
        assert hasattr(self.config,'loss_name'), 'Please check param of loss_name'
        
        if self.config.loss_name in _loss_name:
            if self.config.loss_name == 'MarginLoss':
                return MarginLoss(self.config.margin)
            
            elif self.config.loss_name == 'LogisticLoss':
                return LogisticLoss()
            
            elif self.config.loss_name == 'SigmoidLoss':
                return SigmoidLoss(self.config.adv_temperature)
            
            elif self.config.loss_name == 'selfAdversarialNegativeSamplingLoss':
                return selfAdversarialNegativeSamplingLoss(self.config.margin, self.config.adv_temperature)
        else:
            raise ValueError('model cannot instantiate unsupported loss: {}'.format(self.config.loss_name))

    def scoring_fn(self, h_idx, t_idx, r_idx):
        raise NotImplementedError

    def normalize_param(self):
        raise NotImplementedError
    
    def get_embedding(self):
        raise NotImplementedError