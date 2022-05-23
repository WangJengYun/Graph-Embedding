import torch 
# from sampling import DataLoader
from tqdm.autonotebook import tqdm
from dataloader import DataLoader

class Evaluation(object):
    def __init__(self,model):
        self.model = model
        self.n_entities = model.n_entities
        self.n_relations = model.n_relations
        
        self.distance = model.distance
        self.is_cuda = next(model.parameters()).is_cuda
        if self.is_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.extract_embedding = model.extract_embedding
        self.inference_scoring_function = model.inference_scoring_function
    
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
        
        dataloader = DataLoader(self.knowledge_graph, batch_size = batch_size, device = self.device)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', 
                             desc='Relation prediction evaluation'):

            h_idx, t_idx, r_idx = batch['h'],batch['t'],batch['r']
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
        
        
        if self.device == 'cuda':
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
        
        if self.device == 'cuda':
            self.rank_true_heads = self.rank_true_heads.cuda()
            self.rank_true_tails = self.rank_true_tails.cuda()
            self.filter_rank_true_heads = self.filter_rank_true_heads.cuda()
            self.filter_rank_true_tails = self.filter_rank_true_tails.cuda()

    def evaluate(self, batch_size):
        
        dataloader = DataLoader(self.knowledge_graph, batch_size = batch_size, device = self.device)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', 
                             desc='Link prediction evaluation'):
            
            # batch = list(dataloader)[0]
            h_idx, t_idx, r_idx = batch['h'],batch['t'],batch['r']
            h_emb, t_emb, r_emb, candidates = self.extract_embedding(h_idx, t_idx, r_idx, candidates_type='entity')
            
            # sample_size = h_emb.shape[0]
            # h_emb, t_emb, r_emb, candidates =  h_emb.cpu(), t_emb.cpu(), r_emb.cpu(), candidates.cpu()
            inference_scores = self.inference_scoring_function(candidates, t_emb, r_emb)
            filter_scores = self.filtering_sources(inference_scores, self.dict_of_heads, t_idx, r_idx, h_idx)
            self.rank_true_heads[i * batch_size: (i + 1) * batch_size] = self.get_rank(inference_scores, h_idx).detach()
            self.filter_rank_true_heads[i * batch_size: (i + 1) * batch_size] = self.get_rank(filter_scores, h_idx).detach()

            inference_scores = self.inference_scoring_function(h_emb, candidates, r_emb)
            filter_scores = self.filtering_sources(inference_scores, self.dict_of_tails, h_idx, r_idx, t_idx)
            self.rank_true_tails[i * batch_size: (i + 1) * batch_size] = self.get_rank(inference_scores, t_idx).detach()
            self.filter_rank_true_tails[i * batch_size: (i + 1) * batch_size] = self.get_rank(filter_scores, t_idx).detach()
        
    
    def _mean_rank(self):
        sum_ = (self.rank_true_heads.float().mean() +
                self.rank_true_tails.float().mean()).item()
        filt_sum_ = (self.filter_rank_true_heads.float().mean() +
                    self.filter_rank_true_tails.float().mean()).item()
        return sum_ / 2, filt_sum_ / 2
    
    def _hit_at_k(self, k = 10):
        head_hit = (self.rank_true_heads <= k).float().mean()
        filt_head_hit = (self.filter_rank_true_heads <= k).float().mean()

        tail_hit = (self.rank_true_tails <= k).float().mean()
        filt_tail_hit = (self.filter_rank_true_tails <= k).float().mean()

        return (head_hit + tail_hit)/2 , (filt_head_hit + filt_tail_hit)/2

    def _mrr(self):
        head_mrr = (self.rank_true_heads.float()**(-1)).mean()
        tail_mrr = (self.rank_true_tails.float()**(-1)).mean()
        filt_head_mrr = (self.filter_rank_true_heads.float()**(-1)).mean()
        filt_tail_mrr = (self.filter_rank_true_tails.float()**(-1)).mean()

        return ((head_mrr + tail_mrr).item() / 2,
                (filt_head_mrr + filt_tail_mrr).item() / 2)

    def evaluation_metrics(self):
        metrics = {}
        mean_rank = self._mean_rank()
        hit_at_k = self._hit_at_k(k = 10)
        mrr = self._mrr()
        
        metrics['mead_rank'] = mean_rank[0]
        metrics['mead_rank_by_filter'] = mean_rank[1]

        metrics['hit_at_10'] = hit_at_k[0].item()
        metrics['hit_at_10_by_filter'] = hit_at_k[1].item()

        metrics['mrr'] = mrr[0]
        metrics['mrr_by_filter'] = mrr[1]

        return metrics


if __name__ == "__main__":
    pass 
    # import pandas as pd
    # import torch
    # from data_format import KG_Dataset
    # from sampling import DataLoader
# 
    # df = pd.read_csv('./dataset/FB15k/fb15k_dataset.csv')
    # KG_data = KG_Dataset(input_data = df)
    # KG_train, KG_val, KG_test = KG_data.split_KG(sizes = (483142, 50000, 59071),validation = True)
    # 
    # model = torch.load('./model_files/model_2022.pt')
# 
# 
    # Evaluator = LinkPredictionEvaluator(model, KG_test)
    # Evaluator.evaluate(batch_size = 200)
    # print(Evaluator.mean_rank())
