# from sampling import BernouliNegativeSampler, UniforNegativeSampler
# from torchkge.sampling import BernoulliNegativeSampler, UniformNegativeSampler
from negative_sampling import BernouliNegativeSampler

class DataLoader:
    def __init__(self, kgdataset, batch_size, device, neg_sampling = None):
        self.batch_size = batch_size
        self.device = device

        self.n_sample = kgdataset.n_facts
        self.n_ent = kgdataset.n_ent
        self.n_rel = kgdataset.n_rel
        self.n_batch = self.get_n_batch(self.n_sample, self.batch_size)
        
        self.heads = kgdataset.head_idx
        self.tails = kgdataset.tail_idx
        self.relations = kgdataset.relations
        self.positive_sample = (self.heads, self.relations, self.tails)

        self.sampler = None 
        if neg_sampling == 'BernouliNegativeSampler':
            rel_weight = BernouliNegativeSampler.get_rel_weight(self.heads, self.relations, self.tails)
            #self.sampler = BernouliNegativeSampler(kgdataset,self.n_ent, self.n_rel, self.rel_weight, n_neg = 1)
            self.sampler = BernouliNegativeSampler(rel_weight, self.n_ent, self.n_rel, self.batch_size, self.n_batch, n_neg = 1)        

    @staticmethod 
    def get_n_batch(n_sample, batch_size):
        n_batch = n_sample//batch_size
        if n_sample % n_batch > 0:
            n_batch += 1
        return n_batch

    def __len__(self):        
        return self.n_batch

    def __iter__(self):
        return collate_fn(self.positive_sample, self.sampler, self.n_batch, self.batch_size, self.device)


class collate_fn:
    def __init__(self, positive_sample, sampler, n_batch, batch_size, device):
        self.current_batch = 0 
        
        self.device = device
        self.n_batch = n_batch
        self.batch_size = batch_size

        self.heads, self.relations, self.tails = positive_sample 
        self.sampler = sampler
        if self.sampler:
            self.neg_heads, self.neg_tails = sampler.corrupt_kg(positive_sample)

    def __next__(self):
        if self.current_batch == self.n_batch:
            raise StopIteration
        else :
            i = self.current_batch
            self.current_batch += 1 

            batch = dict()
            batch['h'] = self.heads[(i*self.batch_size):((i+1)*self.batch_size)]
            batch['t'] = self.tails[(i*self.batch_size):((i+1)*self.batch_size)]
            batch['r'] = self.relations[(i*self.batch_size):((i+1)*self.batch_size)]
            if self.sampler:
                batch['nh'] = self.neg_heads[(i*self.batch_size):((i+1)*self.batch_size)]
                batch['nt'] =  self.neg_tails[(i*self.batch_size):((i+1)*self.batch_size)]

            if self.device == 'cuda':
                batch['h'] = batch['h'].cuda()
                batch['t'] = batch['t'].cuda()
                batch['r'] = batch['r'].cuda()
                if self.sampler:
                    batch['nh'] = batch['nh'].cuda()
                    batch['nt'] = batch['nt'].cuda()

            return batch 
    
    def __iter__(self):
        return self 

if __name__ == "__main__":
    import pandas as pd
    from data_format import KG_Dataset
    df = pd.read_csv('./dataset/FB15k/fb15k_dataset.csv')
    KG_data = KG_Dataset(input_data = df)
    kg_train, KG_val, KG_test = KG_data.split_KG(sizes = (483142, 50000, 59071),validation = True)

    traindataloader = DataLoader(kg_train,
                                 neg_sampling = 'BernouliNegativeSampler',
                                 batch_size = 32768, 
                                 device = 'cuda')

    len(traindataloader)

    for i, batch  in enumerate(traindataloader):
        pass