import torch 
from torch import nn
from torch.nn.init import xavier_uniform
from torch.nn.functional import normalize
from model.base_model import Base_Model

class TransE(Base_Model):
    def __init__(self, config, n_entities, n_relations, ent2idx, rel2idx):

        super().__init__(config, n_entities, n_relations)
        
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx

        self.config = config
        self.emb_dim = self.config.emb_dim
    
        self.ent_emb = self.init_embedding(n_entities, self.emb_dim)
        self.rel_emb = self.init_embedding(n_relations, self.emb_dim)
    
    def scoring_fn(self, h_idx, t_idx, r_idx):

        h = normalize(self.ent_emb(h_idx), p=2, dim = 1)
        t = normalize(self.ent_emb(t_idx), p=2, dim = 1)
        r = self.rel_emb(r_idx)

        # return - self.distance(h + r, t)
        return -((h+r)-t).norm(p = 2, dim = -1)**2

    def get_embedding(self):

        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p = 2, dim = 1)

        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p = 2, dim = 1)

    def extract_embedding(self, h_idx, t_idx, r_idx, candidates_type=None):
        
        assert len(h_idx) == len(t_idx)
        assert len(t_idx) == len(r_idx)
        
        b_size = h_idx.shape[0]
        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        if candidates_type:
            candidates = None
            ent_embedding, rel_embedding = self.get_embedding()
            if candidates_type == 'entity':
                candidates = ent_embedding.view(1, self.n_entities, self.emb_dim)
                candidates = candidates.expand(b_size,self.n_entities, self.emb_dim)
        
            elif candidates_type == 'relation':
                candidates = rel_embedding.view(1, self.n_relations, self.emb_dim)
                candidates = candidates.expand(b_size,self.n_relations, self.emb_dim)

            assert candidates != None
            
            return  h, t, r, candidates
        else :
            return h, t, r

    def inference_scoring_function(self,proj_h, proj_t, r):
        
        assert proj_h.shape[0] == proj_t.shape[0]
        assert proj_h.shape[0] == r.shape[0]

        batch_size = proj_h.shape[0]

        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                hr = (proj_h + r).view(batch_size, 1, r.shape[1])
                return - self.distance(hr, proj_t)
            else:
                assert (len(proj_h.shape) == 3) & (len(proj_t.shape) == 2)
                r_ = r.view(batch_size, 1, r.shape[1])
                t_ = proj_t.view(batch_size, 1, r.shape[1])
                return - self.distance(proj_h + r_, t_)
        elif len(r.shape) == 3:
            proj_h = proj_h.view(batch_size, -1, self.emb_dim)
            proj_t = proj_t.view(batch_size, -1, self.emb_dim)
            return - self.distance(proj_h + r, proj_t)