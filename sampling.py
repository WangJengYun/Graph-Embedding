import torch 
from torch import tensor, bernoulli, randint, ones, rand, cat
import pandas as pd 

class _DataLoaderIter:
    def __init__(self, loader):
        self.h = loader.h
        self.t = loader.t
        self.r = loader.r

        self.batch_size = loader.batch_size
        self.use_cuda  = loader.use_cuda

        self.n_batches  = len(loader)
        self.current_batch = 0
    
    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            batch_h = self.h[i * self.batch_size: (i + 1) * self.batch_size]
            batch_t = self.t[i * self.batch_size: (i + 1) * self.batch_size]
            batch_r = self.r[i * self.batch_size: (i + 1) * self.batch_size]

            if self.use_cuda is not None and self.use_cuda == 'batch':
                return batch_h.cuda(), batch_t.cuda(), batch_r.cuda()
            else:
                return batch_h, batch_t, batch_r
    
    def __iter__(self):
        return self 

class DataLoader:
    def __init__(self, kg, batch_size, use_cuda = None):
        self.h = kg.head_idx
        self.t = kg.tail_idx
        self.r = kg.relations

        self.batch_size = batch_size
        self.use_cuda  = use_cuda

        if use_cuda is not None and use_cuda == 'all':
            self.h = kg.head_idx.cuda()
            self.t = kg.tail_idx.cuda()
            self.r = kg.relations.cuda()
    
    def __len__(self):
        n_sample = len(self.h)
        n_batch = n_sample//self.batch_size

        if n_sample % n_batch > 0:
            n_batch += 1
        
        return n_batch
    
    def __iter__(self):
        return _DataLoaderIter(self)

class NegativeSampler:
    
    def __init__(self,kg_train, kg_val = None, kg_test = None, n_neg = 1):
        
        self.kg_train = kg_train
        self.n_ent = kg_train.n_ent
        self.n_sampe = kg_train.n_facts

        self.kg_val = kg_val
        if kg_val is None:
            self.n_sample_val = 0
        else:
            self.n_sample_val = kg_val.n_facts

        self.kg_test = kg_test
        if kg_test is None:
            self.n_sample_test = 0
        else:
            self.n_sample_test = kg_test.n_facts

        self.n_neg = n_neg
    
    def corupt_batch(self, heads, tails, relations, n_neg):
        raise NotImplementedError
    
    def corrupt_kg(self, batch_size, use_cuda, which = 'train'):
        assert which in ['train', 'val', 'test']
        if which == 'val':
            assert self.n_sample_val > 0
        if which == 'test':
            assert self.n_sample_test > 0

        if use_cuda:
            tmp_cuda = 'batch'
        else:
            tmp_cuda = None

        if which == 'train':
            dataloader = DataLoader(self.kg_train, batch_size = batch_size, use_cuda = tmp_cuda)
        elif which == 'val':
            dataloader = DataLoader(self.kg_val, batch_size = batch_size, use_cuda = tmp_cuda)
        else:
            dataloader = DataLoader(self.kg_test, batch_size = batch_size, use_cuda = tmp_cuda)
        
        corr_heads, corr_tails = [], []
        for i, batch in enumerate(dataloader):
            heads, tails, rels = batch[0], batch[1], batch[2]

            neg_heads, neg_tails = self.corrupt_batch(heads, tails, rels, n_neg = 1)

            corr_heads.append(neg_heads)
            corr_tails.append(neg_tails)
        
        if use_cuda:
            return cat(corr_heads).cpu(), cat(corr_tails).cpu()
        else:
            return cat(corr_heads), cat(corr_tails)

class UniforNegativeSampler(NegativeSampler):
    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)
    
    def corrupt_batch(self, heads, tails, relations = None, n_neg = None):
        if n_neg:
            n_neg = self.n_neg
        
        device = heads.device
        assert tails.device == tails.device

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # Randomly choose which samples will have head/tail corrupted 
        # 這裡可能需要換唷~
        mask = torch.bernoulli(torch.ones(size = (batch_size * n_neg,),device = device)).double()

        n_h_cor = int(mask.sum().item())
        neg_heads[mask == 1] = randint(1, self.n_ent, (n_h_cor), device = device)
        neg_tails[mask == 0] = randint(1, self.n_ent, (batch_size * n_neg - n_h_cor), device = device)

        return neg_heads, neg_tails

def get_bernouli_probs(kg):
    
    def get_hpt(t):
        df = pd.DataFrame(t.numpy(),columns=['from','to','rel'])
        df = df.groupby(['rel','to']).count().groupby('rel').mean()
        df.reset_index(inplace = True)
        return {df.loc[i].values[0]:df.loc[i].values[1] for i in df.index}
    
    def get_tph(t):
        df = pd.DataFrame(t.numpy(),columns=['from','to','rel'])
        df = df.groupby(['rel','from']).count().groupby('rel').mean()
        df.reset_index(inplace = True)
        return {df.loc[i].values[0]:df.loc[i].values[1] for i in df.index}

    t = torch.cat((kg.head_idx.view(-1, 1),
                   kg.tail_idx.view(-1, 1),
                   kg.relations.view(-1, 1)), dim = 1)

    hpt = get_hpt(t)
    tph = get_tph(t)

    assert hpt.keys() == tph.keys()

    for k in tph.keys():
        tph[k] = tph[k]/(hpt[k]+tph[k])

    return tph

class BernouliNegativeSampler(NegativeSampler):
    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)
        self.bern_probs = self.evaluate_probilities()

    def evaluate_probilities(self):
        bern_probs = get_bernouli_probs(self.kg_train)

        tmp = []
        for i in range(self.kg_train.n_rel):
            if i in bern_probs.keys():
                tmp.append(bern_probs[i])
            else:
                tmp.append(0.5)
        
        return torch.tensor(tmp).float()
    
    def corrupt_batch(self, heads, tails, relations = None, n_neg = None):
        if n_neg:
            n_neg = self.n_neg
        
        device = heads.device
        assert (tails.device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # Randomly choose which samples will have head/tail corrupted 
        # 這裡可能需要換唷~
        
        mask = torch.bernoulli(self.bern_probs[relations.type(torch.long)].repeat(n_neg)).double()

        n_h_cor = int(mask.sum().item())

        neg_heads[mask == 1] = randint(1, self.n_ent, (n_h_cor,), device = device)
        neg_tails[mask == 0] = randint(1, self.n_ent, (batch_size * n_neg - n_h_cor,), device = device)

        return neg_heads, neg_tails



if __name__ == '__main__':
    from data_format import KG_Dataset
    import pandas as pd 
    df = pd.read_csv('./dataset/FB15k/freebase_mtr100_mte100-test.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    
    KG_data = KG_Dataset(input_data = df)
    KG_train, KG_val, KG_test = KG_data.split_KG(share=0.8,validation=True)

    dataloader = DataLoader(KG_test, batch_size= 32, use_cuda='batch')