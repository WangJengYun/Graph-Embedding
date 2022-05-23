import torch 
from torch import nn
from torch.nn.init import xavier_uniform
from torch.nn.functional import normalize
from model.base_model import Base_Model

class RotatE(Base_Model):

    def __init__(self,config, n_entities, n_relations, ent2idx, rel2idx):
        super().__init__(config, n_entities, n_relations)

        self.ent2idx = ent2idx
        self.rel2idx = rel2idx

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.config = config
        self.margin = self.config.margin
        self.epsilon = self.config.epsilon
        self.emb_dim = self.config.emb_dim
        self.cal_mode = self.config.cal_mode
        self.ent_emb_dim = self.config.emb_dim*2
        self.rel_emb_dim = self.config.emb_dim

        self.pi = 3.14159265359
        
        self.ent_emb = nn.Embedding(n_entities, self.ent_emb_dim)
        self.rel_emb = nn.Embedding(n_entities, self.rel_emb_dim)
        
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon)/self.ent_emb_dim])
        )
        
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon)/self.rel_emb_dim])
        )
        nn.init.uniform_(
            tensor = self.ent_emb.weight.data,
            a = - self.ent_embedding_range.item(),
            b = self.ent_embedding_range.item()
        )

        nn.init.uniform_(
            tensor = self.rel_emb.weight.data,
            a = - self.rel_embedding_range.item(),
            b = self.rel_embedding_range.item()
        )

        self.margin = nn.Parameter(torch.Tensor([self.config.margin]))
        self.margin.requires_grad = False

    def scoring_fn(self, h_idx, t_idx, r_idx):
        
        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        re_h, im_h = torch.chunk(h, 2 , dim = 1)
        re_t, im_t = torch.chunk(t, 2 , dim = 1)

        phase_r = r / (self.rel_embedding_range.item() / self.pi)
        re_r = torch.cos(phase_r)
        im_r = torch.sin(phase_r)

        re_h = re_h.view(re_r.shape[0],1, self.emb_dim)
        im_h = im_h.view(re_r.shape[0],1, self.emb_dim)
        re_t = re_t.view(re_r.shape[0],1, self.emb_dim)
        im_t = im_t.view(re_r.shape[0],1, self.emb_dim)
        re_r = re_r.view(re_r.shape[0],1, self.emb_dim)
        im_r = im_r.view(re_r.shape[0],1, self.emb_dim)

        if self.cal_mode == 'head_batch':
            re_score = re_r * re_t + im_r * im_r
            im_score = re_r * im_t + im_r * re_r
            re_score = re_score - re_h
            im_score = im_score - re_h
        else : 
            re_score = re_h * re_r - im_h * im_r
            im_score = re_h * im_r + im_h * re_r
            re_score = re_score - re_t
            im_score = im_score - re_t

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0, p = 1).sum(dim = -1)

        return self.margin - score.flatten()
    
    def normalize_parameters(self):
        pass
    
    def extract_embedding(self, h_idx, t_idx, r_idx, candidates_type=None): 

        assert len(h_idx) == len(t_idx)
        assert len(t_idx) == len(r_idx)
        
        batch_size = h_idx.shape[0]

        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)
        
        re_h, im_h = torch.chunk(h, 2 , dim = 1)
        re_t, im_t = torch.chunk(t, 2 , dim = 1)

        phase_r = r / (self.rel_embedding_range.item() / self.pi)
        re_r = torch.cos(phase_r)
        im_r = torch.sin(phase_r)


        if candidates_type:            
            if candidates_type == 'entity':
                re_ent_emb,im_ent_emb = torch.chunk(self.ent_emb.weight.data, 2, dim = 1)

                re_candidates = re_ent_emb.view(1,  self.n_entities, self.emb_dim)
                re_candidates = re_candidates.expand(batch_size,  self.n_entities, self.emb_dim)

                im_candidates =im_ent_emb.view(1,  self.n_entities, self.emb_dim)
                im_candidates = im_candidates.expand(batch_size,  self.n_entities, self.emb_dim)
            else:
                candidates = self.rel_emb.weight.data.view(1, self.n_relations, self.emb_dim)
                candidates = candidates.expand(batch_size, self.n_relations, self.emb_dim)
                phase_candidates = candidates / (self.rel_embedding_range.item() / self.pi)
                re_candidates = torch.cos(phase_candidates)
                im_candidates = torch.sin(phase_candidates)

            return (re_h, im_h), (re_t, im_t), (re_r, im_r), (re_candidates, im_candidates)

        return (re_h, im_h), (re_t, im_t), (re_r, im_r)

    def inference_scoring_function(self,h, t, r):
        
        re_h, im_h = h[0], h[1]
        re_t, im_t = t[0], t[1]
        re_r, im_r = r[0], r[1]
        b_size = re_h.shape[0]
        
        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            re_h = re_h.view(re_r.shape[0],1, self.emb_dim)
            im_h = im_h.view(re_r.shape[0],1, self.emb_dim)
            re_r = re_r.view(re_r.shape[0],1, self.emb_dim)
            im_r = im_r.view(re_r.shape[0],1, self.emb_dim)

            re_score = re_h * re_r - im_h * im_r
            im_score = re_h * im_r + im_h * re_r
            re_score = re_score - re_t
            im_score = im_score - re_t

            score = torch.stack([re_score, im_score], dim = 0)
            score = score.norm(dim = 0).sum(dim = -1)

            return (self.margin - score)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)
            re_t = re_t.view(re_r.shape[0],1, self.emb_dim)
            im_t = im_t.view(re_r.shape[0],1, self.emb_dim)
            re_r = re_r.view(re_r.shape[0],1, self.emb_dim)
            im_r = im_r.view(re_r.shape[0],1, self.emb_dim)

            re_score = re_r * re_t + im_r * im_r
            im_score = re_r * im_t + im_r * re_r
            re_score = re_score - re_h
            im_score = im_score - re_h

            score = torch.stack([re_score, im_score], dim = 0)
            score = score.norm(dim = 0).sum(dim = -1)

            return (self.margin - score)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            re_h = re_h.view(re_r.shape[0],1, self.emb_dim)
            im_h = im_h.view(re_r.shape[0],1, self.emb_dim)
            re_t = re_t.view(re_r.shape[0],1, self.emb_dim)
            im_t = im_t.view(re_r.shape[0],1, self.emb_dim)

            re_score = re_h * re_r - im_h * im_r
            im_score = re_h * im_r + im_h * re_r
            re_score = re_score - re_t
            im_score = im_score - re_t

            score = torch.stack([re_score, im_score], dim = 0)
            score = score.norm(dim = 0).sum(dim = -1)

            return (self.margin - score)