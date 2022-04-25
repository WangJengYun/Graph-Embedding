from model.base import TranslationModel

from torch import nn
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform


class TransE(TranslationModel):
    def __init__(self, config, n_entities, n_relations):

        super().__init__(config, n_entities, n_relations)
        self.config = config
        self.emb_dim = self.config.emb_dim
        self.ent_emb = self.init_embedding(n_entities, self.emb_dim)
        self.rel_emb = self.init_embedding(n_relations, self.emb_dim)

        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p = 2, dim = 1)
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p = 2, dim = 1)
    
    @staticmethod 
    def init_embedding(n_vector, dim):

        entity_embeddings = nn.Embedding(n_vector, dim)
        xavier_uniform(entity_embeddings.weight.data)

        return entity_embeddings

    def get_embedding(self):

        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p = 2, dim = 1)

        return self.ent_emb.weight.data, self.rel_emb.weight.data
    
    def scoring_fn(self, h_idx, t_idx, r_idx):

        h = normalize(self.ent_emb(h_idx), p=2, dim = 1)
        t = normalize(self.ent_emb(t_idx), p=2, dim = 1)
        r = self.rel_emb(r_idx)

        return - self.distance(h + r, t)

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p = 2, dim = 1)