import torch 
from sampling import DataLoader
from tqdm.autonotebook import tqdm

class Evaluation(object):
    def __init__(self,model):
        self.model = model
        self.n_entities = model.n_entities
        self.n_relations = model.n_relations
        self.ent_emb = model.ent_emb
        self.rel_emb = model.rel_emb
        self.emb_dim =  model.emb_dim
        
        self.distance = model.distance
        self.use_cuda = next(model.parameters()).is_cuda
        self.embedding = model.get_embedding()

    def extract_embedding(self, h_idx, t_idx, r_idx, candidates_type='entity'):
        
        h_size = h_idx.shape[0]
        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        ent_embedding, rel_embedding = self.embedding

        if candidates_type == 'entity':
            candidates = ent_embedding.view(1, self.n_entities, self.emb_dim)
            candidates = candidates.expand(h_size,self.n_entities, self.emb_dim)
        
        elif candidates_type == 'relation':
            candidates = rel_embedding.view(1, self.n_relations, self.emb_dim)
            candidates = candidates.expand(h_size,self.n_relations, self.emb_dim)
        else:
            pass 

        return h, t, r, candidates
    

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
    
    def evaluate(self, knowledge_graph, batch_size):
        raise NotImplementedError 

    @staticmethod
    def get_rank(scores,true_idx,low_values = False):
        threshold_scores = scores.gather(1, true_idx.long().view(-1, 1))
        if low_values:
            return (scores <= threshold_scores).sum(dim=1)
        else:
            return (scores >= threshold_scores).sum(dim=1)


    @staticmethod
    def get_true_target(dict_of_element,key_idx1,key_idx2,true_idx):
        try:
            true_targets = dict_of_element[key_idx1,key_idx2].copy()
            if true_idx is not None:
                true_targets.remove(true_idx)
                if len(true_targets) > 0:
                    return torch.tensor(list(true_targets)).long()
                else:
                    return None
            else:
                return  torch.tensor(list(true_targets)).long()
        except KeyError:
            return None
    
    def filtering_sources(self,scores, dict_of_element, key_1, key_2, target):
        filter_scores = scores.clone()
        sample_size = scores.shape[0]
        
        for b in range(sample_size):
            input_key_1 = key_1[b].item()
            input_key_2 = key_2[b].item()
            input_target = target[b].item()
            true_targets  = self.get_true_target(dict_of_element = dict_of_element,
                                                key_idx1 = input_key_1, 
                                                key_idx2 = input_key_2, 
                                                true_idx = input_target)
            if true_targets is None:
                continue
            
            filter_scores[b][true_targets] = -float('Inf')
        
        return filter_scores


class RelationPredictionEvaluator(Evaluation):
    def __init__(self,model, knowledge_graph, directed=True):
        super().__init__(model)
        self.knowledge_graph = knowledge_graph
        self.dict_of_rels = knowledge_graph.dict_of_rels
        self.rank_true_rels = torch.empty(size=(knowledge_graph.n_facts,)).long()
        self.filter_rank_true_rels = torch.empty(size=(knowledge_graph.n_facts,)).long()

        if self.use_cuda:
            self.rank_true_rels = self.rank_true_rels.cuda()
            self.rank_true_rels = self.rank_true_rels.cuda()


    def evaluate(self, batch_size):
        
        dataloader = DataLoader(self.knowledge_graph, batch_size = batch_size, use_cuda = 'batch')

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', 
                             desc='Relation prediction evaluation'):

            h_idx, t_idx, r_idx = batch
            h_emb, t_emb, r_emb, candidates = self.extract_embedding(h_idx, t_idx, r_idx, candidates_type='relation')
            inference_scores = self.inference_scoring_function(h_emb, t_emb, candidates)

            filter_scores = inference_scores.clone()
            sample_size = h_emb.shape[0]
            for b in range(sample_size):

                input_h_idx = h_idx[b].item()
                input_t_idx = t_idx[b].item()
                input_r_idx = r_idx[b].item()

                true_targets  = self.get_true_target(dict_of_element = self.dict_of_rels,
                                                    key_idx1 = input_h_idx, 
                                                    key_idx2 = input_t_idx, 
                                                    true_idx = input_r_idx)
                if true_targets is None:
                    continue
                
                filter_scores[b][true_targets] = - float('Inf')
        
            self.rank_true_rels[i * sample_size: (i + 1) * sample_size] = self.get_rank(inference_scores, r_idx).detach()
            self.filter_rank_true_rels[i * sample_size: (i + 1) * sample_size] = self.get_rank(filter_scores, r_idx).detach()
        
        
        if self.use_cuda:
            self.rank_true_rels = self.rank_true_rels.cpu()
            self.filter_rank_true_rels = self.filter_rank_true_rels.cpu()
    
    def mean_rank(self):
        sum_ = self.rank_true_rels.float().mean().item()
        filt_sum_ = self.filter_rank_true_rels.float().mean().item()
        return sum_, filt_sum_

class LinkPredictionEvaluator(Evaluation):
    def __init__(self, model, knowledge_graph):
        super().__init__(model)
        
        self.knowledge_graph = knowledge_graph
        self.dict_of_heads = knowledge_graph.dict_of_heads
        self.dict_of_tails = knowledge_graph.dict_of_tails

        self.rank_true_heads = torch.empty(size=(knowledge_graph.n_facts,)).long()
        self.rank_true_tails = torch.empty(size=(knowledge_graph.n_facts,)).long()
        self.filter_rank_true_heads = torch.empty(size=(knowledge_graph.n_facts,)).long()
        self.filter_rank_true_tails = torch.empty(size=(knowledge_graph.n_facts,)).long()
        
        if self.use_cuda:
            self.rank_true_heads = self.rank_true_heads.cuda()
            self.rank_true_tails = self.rank_true_tails.cuda()
            self.filter_rank_true_heads = self.filter_rank_true_heads.cuda()
            self.filter_rank_true_tails = self.filter_rank_true_tails.cuda()

    def evaluate(self, batch_size):
        
        dataloader = DataLoader(self.knowledge_graph, batch_size = batch_size, use_cuda = 'batch')

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', 
                             desc='Link prediction evaluation'):
            
            # batch = list(dataloader)[0]
            h_idx, t_idx, r_idx = batch
            h_emb, t_emb, r_emb, candidates = self.extract_embedding(h_idx, t_idx, r_idx, candidates_type='entity')
            
            sample_size = h_emb.shape[0]
            
            inference_scores = self.inference_scoring_function(candidates, t_emb, r_emb)
            filter_scores = self.filtering_sources(inference_scores, self.dict_of_heads, t_idx, r_idx, h_idx)
            self.rank_true_heads[i * batch_size: (i + 1) * batch_size] = self.get_rank(inference_scores, h_idx).detach()
            self.filter_rank_true_heads[i * batch_size: (i + 1) * batch_size] = self.get_rank(filter_scores, h_idx).detach()

            inference_scores = self.inference_scoring_function(h_emb, candidates, r_emb)
            filter_scores = self.filtering_sources(inference_scores, self.dict_of_tails, h_idx, r_idx, t_idx)
            self.rank_true_tails[i * batch_size: (i + 1) * batch_size] = self.get_rank(inference_scores, t_idx).detach()
            self.filter_rank_true_tails[i * batch_size: (i + 1) * batch_size] = self.get_rank(filter_scores, t_idx).detach()
        
        if self.use_cuda:
            
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filter_rank_true_heads = self.filter_rank_true_heads.cpu()
            self.filter_rank_true_tails = self.filter_rank_true_tails.cpu()
    
    def mean_rank(self):
        sum_ = (self.rank_true_heads.float().mean() +
                self.rank_true_tails.float().mean()).item()
        filt_sum_ = (self.filter_rank_true_heads.float().mean() +
                    self.filter_rank_true_tails.float().mean()).item()
        return sum_ / 2, filt_sum_ / 2


if __name__ == "__main__":
    import pandas as pd
    import torch
    from data_format import KG_Dataset
    from sampling import DataLoader

    df = pd.read_csv('./dataset/FB15k/fb15k_dataset.csv')
    KG_data = KG_Dataset(input_data = df)
    KG_train, KG_val, KG_test = KG_data.split_KG(sizes = (483142, 50000, 59071),validation = True)
    
    model = torch.load('./model_files/model_2022.pt')


    Evaluator = LinkPredictionEvaluator(model, KG_test)
    Evaluator.evaluate(batch_size = 200)
    print(Evaluator.mean_rank())
