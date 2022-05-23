import torch 
from torch import nn
from torch.nn.init import xavier_uniform
from torch.nn.functional import normalize
from model.base_model import Base_Model

class ComplEx(Base_Model):
    def __init__(self,config, n_entities, n_relations, ent2idx, rel2idx):

        super().__init__(config, n_entities, n_relations)
        self.ent2idx = ent2idx
        self.rel2idx = rel2idx        
        
        self.config = config
        self.emb_dim = self.config.emb_dim
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.re_ent_emb = self.init_embedding(n_entities, self.emb_dim)
        self.im_ent_emb = self.init_embedding(n_entities, self.emb_dim)
        self.re_rel_emb = self.init_embedding(n_relations, self.emb_dim)
        self.im_rel_emb = self.init_embedding(n_relations, self.emb_dim)

    def scoring_fn(self, h_idx, t_idx, r_idx):
        
        re_h, im_h = self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        re_t, im_t = self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        re_r, im_r = self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        return (re_h * (re_r * re_t + im_r * im_t) + im_h * (
                    re_r * im_t - im_r * re_t)).sum(dim=1)

    def normalize_parameters(self):
        pass 
    
    def get_embeddings(self):
        return self.re_ent_emb.weight.data, self.im_ent_emb.weight.data,\
                self.re_rel_emb.weight.data, self.im_rel_emb.weight.data      

    def extract_embedding(self, h_idx, t_idx, r_idx, candidates_type=None): 

        assert len(h_idx) == len(t_idx)
        assert len(t_idx) == len(r_idx)
        
        batch_size = h_idx.shape[0]

        re_h, im_h = self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        re_t, im_t = self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        re_r, im_r = self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        if candidates_type:
            
            if candidates_type == 'entity':
                re_candidates = self.re_ent_emb.weight.data.view(1,  self.n_entities, self.emb_dim)
                re_candidates = re_candidates.expand(batch_size,  self.n_entities, self.emb_dim)

                im_candidates = self.im_ent_emb.weight.data.view(1,  self.n_entities, self.emb_dim)
                im_candidates = im_candidates.expand(batch_size,  self.n_entities, self.emb_dim)
            else:
                re_candidates = self.re_rel_emb.weight.data.view(1, self.n_relations, self.emb_dim)
                re_candidates = re_candidates.expand(batch_size, self.n_relations, self.emb_dim)

                im_candidates = self.im_rel_emb.weight.data.view(1, self.n_relations, self.emb_dim)
                im_candidates = im_candidates.expand(batch_size, self.n_relations, self.emb_dim)

            return (re_h, im_h), (re_t, im_t), (re_r, im_r), (re_candidates, im_candidates)

        return (re_h, im_h), (re_t, im_t), (re_r, im_r)
    
    def inference_scoring_function(self,h, t, r):
        
        re_h, im_h = h[0], h[1]
        re_t, im_t = t[0], t[1]
        re_r, im_r = r[0], r[1]
        b_size = re_h.shape[0]

        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            return ((re_h * re_r - im_h * im_r).view(b_size, 1, self.emb_dim) * re_t
                    + (re_h * im_r + im_h * re_r).view(b_size, 1, self.emb_dim) * im_t).sum(dim=2)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)

            return (re_h * (re_r * re_t + im_r * im_t).view(b_size, 1, self.emb_dim)
                    + im_h * (re_r * im_t - im_r * re_t).view(b_size, 1, self.emb_dim)).sum(dim=2)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            return ((re_h * re_t + im_h * im_t).view(b_size, 1, self.emb_dim) * re_r
                    + (re_h * im_t - im_h * re_t).view(b_size, 1, self.emb_dim) * im_r).sum(dim=2) 