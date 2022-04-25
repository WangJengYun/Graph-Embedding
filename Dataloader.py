from sampling import BernouliNegativeSampler, UniforNegativeSampler
# from torchkge.sampling import BernoulliNegativeSampler, UniformNegativeSampler

class NegativeSampling:
    def __init__(self, dataloader):
        self.h = dataloader.h
        self.t = dataloader.t
        self.r = dataloader.r

        self.nh, self.nt = dataloader.sampler.corrupt_kg(dataloader.batch_size,
                                                         dataloader.tmp_cuda)

        self.use_cuda = dataloader.use_cuda
        self.batch_size = dataloader.batch_size
        
        n_sample = len(self.h)
        self.n_batch = n_sample//self.batch_size

        if n_sample % self.n_batch > 0:
            self.n_batch += 1
        
        if self.use_cuda:
            self.nh = self.nh.cuda()
            self.nt = self.nt.cuda()
        self.current_batch = 0 

    def __next__(self):
        if self.current_batch == self.n_batch:
            raise StopIteration
        else :
            i = self.current_batch
            self.current_batch += 1 

            batch = dict()
            batch['h'] = self.h[(i*self.batch_size):((i+1)*self.batch_size)]
            batch['t'] = self.t[(i*self.batch_size):((i+1)*self.batch_size)]
            batch['r'] = self.r[(i*self.batch_size):((i+1)*self.batch_size)]
            batch['nh'] = self.nh[(i*self.batch_size):((i+1)*self.batch_size)]
            batch['nt'] = self.nt[(i*self.batch_size):((i+1)*self.batch_size)]

            if self.use_cuda == 'batch':
                batch['h'] = batch['h'].cuda()
                batch['t'] = batch['t'].cuda()
                batch['r'] = batch['r'].cuda()

                batch['nh'] = batch['nh'].cuda()
                batch['nt'] = batch['nt'].cuda()

            return batch 
    
    def __iter(self):
        return self 

class TrainDataLoader:
    def __init__(self, kg, batch_size, sampling_type, use_cuda = None):

        self.h = kg.head_idx
        self.t = kg.tail_idx
        self.r = kg.relations

        self.use_cuda = use_cuda
        self.batch_size = batch_size

        if sampling_type == 'unif':
            self.sampler = UniforNegativeSampler(kg)
        elif sampling_type == 'bern':
            self.sampler = BernouliNegativeSampler(kg)

        self.tmp_cuda  = use_cuda in ['batch','all']

        if use_cuda is not None and use_cuda == 'all':
            self.h = self.h.cuda()
            self.t = self.t.cuda()
            self.r = self.r.cuda()

    def __len__(self):
        n_sample = len(self.h)
        n_batch = n_sample//self.batch_size

        if n_sample % n_batch > 0:
            n_batch += 1
        
        return n_batch

    def __iter__(self):
        return NegativeSampling(self)