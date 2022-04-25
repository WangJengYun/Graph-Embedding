from torch.nn import Module
from scoring_fuctiuon import l1_distance, l2_distance

class Base_Model(Module):
    def __init__(self, n_entities, n_relations):
        super().__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
    
    def forward(self, heads, tails, relations,
                      negative_head, negative_tails, negative_relations = None):
        pos = self.scoring_fn(heads, tails, relations)

        if negative_relations is None:
            negative_relations = relations

        if negative_head.shape[0] > negative_relations.shape[0]:

            n_neg = int(negative_head.shape[0]/negative_relations.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_fn(negative_head,
                                  negative_tails,
                                  negative_relations.repeat(n_neg))
        else:
            neg = self.scoring_fn(negative_head,
                                  negative_tails,
                                  negative_relations)
        return pos, neg 

    def scoring_fn(self, h_idx, t_idx, r_idx):
        raise NotImplementedError

    def normalize_param(self):
        raise NotImplementedError
    
    def get_embedding(self):
        raise NotImplementedError

class TranslationModel(Base_Model):
    def __init__(self, config, n_entities, n_relations):
        super().__init__(n_entities, n_relations)
        assert config.distance_type in ['L1','L2'], 'Please check the type of distance'

        if config.distance_type == 'L1':
            self.distance = l1_distance
        elif config.distance_type == 'L2':
            self.distance = l2_distance
        else:
            raise ValueError
    
    def scoring_fn(self, h_idx, t_idx, r_idx):
        raise NotImplementedError

    # def normalize_param(self):
    #     raise NotImplementedError
    
    def get_embedding(self):
        raise NotImplementedError
    
if __name__ == '__main__':
    pass