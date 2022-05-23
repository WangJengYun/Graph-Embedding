import torch 
from torch import tensor, bernoulli, randint, ones, rand, cat
import pandas as pd 

class NegativeSampler:
    
    def __init__(self, batch_size, n_batch):

        self.batch_size = batch_size
        self.n_batch = n_batch
    
    def corupt_batch(self, heads, tails, relations):
        raise NotImplementedError
    
    def corrupt_kg(self, positive_sample):

        heads, relations, tails = positive_sample 
        
        neg_heads, neg_tails = [],[]
        for i in range(self.n_batch):
            batch_heads = heads[i * self.batch_size: (i + 1) * self.batch_size]
            batch_tails = tails[i * self.batch_size: (i + 1) * self.batch_size]
            batch_relations = relations[i * self.batch_size: (i + 1) * self.batch_size]

            nh, nt = self.corrupt_batch(batch_heads, batch_tails, batch_relations)
            
            neg_heads.append(nh)
            neg_tails.append(nt)

        return torch.cat(neg_heads), torch.cat(neg_tails)


class BernouliNegativeSampler(NegativeSampler):
    def __init__(self,rel_weight, n_ent, n_rel, batch_size, n_batch, n_neg):
        super().__init__(batch_size, n_batch)
        self.rel_weight = rel_weight
        self.n_rel = n_rel
        self.n_ent = n_ent
        self.batch_size = batch_size
        self.n_batch = n_batch
        self.n_neg = n_neg

        self.bern_probs = self.evaluate_probilities()

    def evaluate_probilities(self):
        tmp = []
        for i in range(self.n_rel):
            if i in self.rel_weight.keys():
                tmp.append(self.rel_weight[i])
            else:
                tmp.append(0.5)
        
        return torch.tensor(tmp).float()
    
    def corrupt_batch(self, batch_heads, batch_tails, batch_relations):
        
        device = batch_heads.device
        assert (batch_heads.device == batch_tails.device)

        current_batch_size = batch_heads.shape[0]
        batch_neg_heads = batch_heads.repeat( self.n_neg)
        batch_neg_tails = batch_tails.repeat( self.n_neg)

        # Randomly choose which samples will have head/tail corrupted        
        mask = torch.bernoulli(self.bern_probs[batch_relations.type(torch.long)].repeat(self.n_neg)).double()
        
        n_h_cor = int(mask.sum().item())
        batch_neg_heads[mask == 1] = randint(1, self.n_ent, (n_h_cor,), device = device)
        batch_neg_tails[mask == 0] = randint(1, self.n_ent, (current_batch_size *  self.n_neg - n_h_cor,), device = device)

        return batch_neg_heads, batch_neg_tails

    @staticmethod
    def get_rel_weight(head, relations, tail):

        def get_hpt(triples):
            df = pd.DataFrame(triples.numpy(),columns=['from','to','rel'])
            df = df.groupby(['rel','to']).count().groupby('rel').mean()
            df.reset_index(inplace = True)
            return {df.loc[i].values[0]:df.loc[i].values[1] for i in df.index}

        def get_tph(t):
            df = pd.DataFrame(triples.numpy(),columns=['from','to','rel'])
            df = df.groupby(['rel','from']).count().groupby('rel').mean()
            df.reset_index(inplace = True)
            return {df.loc[i].values[0]:df.loc[i].values[1] for i in df.index}

        triples = torch.cat((head.view(-1, 1), tail.view(-1, 1),relations.view(-1, 1)), dim = 1)


        hpt = get_hpt(triples)
        tph = get_tph(triples)

        assert hpt.keys() == tph.keys()

        rel_weight = dict()
        for k in tph.keys():
            rel_weight[k] = tph[k]/(hpt[k]+tph[k])

        return rel_weight



if __name__ == '__main__':
    pass

