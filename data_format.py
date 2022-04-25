import torch
from torch.utils.data import Dataset

import pandas as pd 
from collections import defaultdict

from utils.exceptions import DataInfoError, WrongArgumentsError

class KG_Dataset(Dataset):

    def __init__(self, input_data = None,
                       ent2idx = None,
                       rel2idx = None,
                       dict_of_heads = None,
                       dict_of_tails = None,
                       dict_of_rels = None):
        
        if ent2idx is None :
            self.ent2idx = self.get_element(input_data, elementtype='from') 
        else:
            self.ent2idx = ent2idx

        if rel2idx is None:
            self.rel2idx = self.get_element(input_data, elementtype='rel') 
        else:
            self.rel2idx = rel2idx

        self.n_ent = len(self.ent2idx)
        self.n_rel = len(self.rel2idx)
        if type(input_data) == pd.DataFrame:
            assert ('from' in input_data.columns) &  ('rel' in input_data.columns) &  ('to' in input_data.columns) 

            self.n_facts = len(input_data) 
            self.head_idx = torch.tensor(input_data['from'].map(self.ent2idx).tolist())
            self.tail_idx = torch.tensor(input_data['to'].map(self.ent2idx).tolist())
            self.relations = torch.tensor(input_data['rel'].map(self.rel2idx).tolist())
        elif type(input_data) == dict:
            assert ('from' in input_data.keys()) &  ('rel' in input_data.keys()) &  ('to' in input_data.keys()) 
            
            self.n_facts = input_data['rel'].shape[0]
            self.head_idx = input_data['from']
            self.tail_idx = input_data['to']
            self.relations = input_data['rel']
        else :
            raise DataInfoError("Please check the arguments(input_data).")

        if dict_of_heads is None or dict_of_tails is None or dict_of_rels is None:
            self.dict_of_heads = defaultdict(set)
            self.dict_of_tails = defaultdict(set)
            self.dict_of_rels = defaultdict(set)
            self.evaluate_dicts()
        else :
            self.dict_of_heads = dict_of_heads
            self.dict_of_tails = dict_of_tails
            self.dict_of_rels = dict_of_rels


        try :
            self.datacheck()
        except AssertionError:
            raise DataInfoError("Please check the information of data.")

    def __len__(self):
        return self.n_facts
    
    def __getitem__(self,item):
        return (self.head_idx[item].item(),
                self.tail_idx[item].item(),
                self.relations[item].item())
    
    def datacheck(self):
        
        assert (type(self.dict_of_heads) == defaultdict) & (type(self.dict_of_tails) == defaultdict)
        assert (type(self.ent2idx) == dict) & (type(self.rel2idx) == dict)
        assert (len(self.ent2idx) == self.n_ent) & (len(self.rel2idx) == self.n_rel)
        assert len(self.relations) == len(self.tail_idx) == len(self.head_idx)
        assert (type(self.head_idx) == torch.Tensor) & (type(self.tail_idx) == torch.Tensor) & (type(self.relations) == torch.Tensor)
    
    def split_KG(self, share = 0.8, sizes = None, validation = False):

        if sizes:
            if len(sizes) == 2:
                try:
                   assert sizes[0] + sizes[1] == self.n_facts
                except AssertionError:
                    raise WrongArgumentsError('Please check the arguments(sizes should sum to the number of samples.).')
            elif len(sizes) == 3:
                try:
                    assert sizes[0] + sizes[1] + sizes[2] == self.n_facts
                except AssertionError:
                    raise WrongArgumentsError('Please check the arguments(sizes should sum to the number of samples.).')
            else:
                raise WrongArgumentsError('Please check the arguments(the lenght of sizes must be less than 3 and more than 1).')        
        else:
            share <= 1
        
        if sizes is not None:
            if (len(sizes) == 3):
                if validation:
                    mask_tran = torch.cat((torch.tensor([1 for _ in range(sizes[0])]),
                                           torch.tensor([0 for _ in range(sizes[1]+sizes[2])]))).bool()
                    mask_val = torch.cat((torch.tensor([0 for _ in range(sizes[0])]),
                                          torch.tensor([1 for _ in range(sizes[1])]),
                                          torch.tensor([0 for _ in range(sizes[2])]))).bool()
                    mask_test = ~(mask_tran|mask_val)
                else:
                    mask_tran = torch.cat((torch.tensor([1 for _ in range(sizes[0])]),
                                           torch.tensor([0 for _ in range(sizes[1]+sizes[2])]))).bool()
                    mask_val = torch.cat((torch.tensor([0 for _ in range(sizes[0])]),
                                          torch.tensor([1 for _ in range(sizes[1])]),
                                          torch.tensor([0 for _ in range(sizes[2])]))).bool()
                    mask_test = ~(mask_tran|mask_val)
            else:
                mask_tran = torch.cat((torch.tensor([1 for _ in range(sizes[0])]),
                                       torch.tensor([0 for _ in range(sizes[1])]))).bool()
                mask_test = ~(mask_tran)
        else:
            if validation:
                mask_tran, mask_val, mask_test = self.get_mask(share,validation=True)
            else:
                mask_tran, mask_test = self.get_mask(share,validation=False)
        if validation:
            return (KG_Dataset(
                        input_data={'from': self.head_idx[mask_tran],
                                    'rel': self.relations[mask_tran],
                                    'to': self.tail_idx[mask_tran]},
                        ent2idx=self.ent2idx, rel2idx=self.rel2idx,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    KG_Dataset(
                        input_data={'from': self.head_idx[mask_val],
                                    'rel': self.relations[mask_val],
                                    'to': self.tail_idx[mask_val]},
                        ent2idx=self.ent2idx, rel2idx=self.rel2idx,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                   KG_Dataset(
                        input_data={'from': self.head_idx[mask_test],
                                    'rel': self.relations[mask_test],
                                    'to': self.tail_idx[mask_test]},
                        ent2idx=self.ent2idx, rel2idx=self.rel2idx,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels))
        else:
            return (KG_Dataset(
                        input_data={'from': self.head_idx[mask_tran],
                                    'rel': self.relations[mask_tran],
                                    'to': self.tail_idx[mask_tran]},
                        ent2idx=self.ent2idx, rel2idx=self.rel2idx,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels),
                    KG_Dataset(
                        input_data={'from': self.head_idx[mask_test],
                                    'rel': self.relations[mask_test],
                                    'to': self.tail_idx[mask_test]},
                        ent2idx=self.ent2idx, rel2idx=self.rel2idx,
                        dict_of_heads=self.dict_of_heads,
                        dict_of_tails=self.dict_of_tails,
                        dict_of_rels=self.dict_of_rels))

    def get_mask(self,share,validation = False):

        unique_r, count_r = self.relations.unique(return_counts = True)
        unique_e, count_e = torch.cat((self.head_idx,self.tail_idx)).unique(return_counts = True)

        mask_train = torch.zeros_like(self.relations).bool()
        if validation :
            mask_val = torch.zeros_like(self.relations).bool()

        for i, r in enumerate(unique_r):
            rand = torch.randperm(count_r[i].item())

            sub_mask = torch.eq(self.relations, r).nonzero(as_tuple=False)[:,0]
            assert len(sub_mask) == count_r[i].item()

            if validation:
                train_size, val_size, test_size = self.get_size(count_r[i].item(),
                                                           share = share,
                                                           validation = True)

                mask_train[sub_mask[:train_size]] = True
                mask_val[sub_mask[train_size:train_size+val_size]] = True
            else:
                train_size, test_size = self.get_size(count_r[i].item(),
                                                 share=share,
                                                 validation=False)     
                mask_train[sub_mask[:train_size]] = True     

        nonmissing_e = torch.cat((self.head_idx[mask_train],self.tail_idx[mask_train])).unique()

        if len(unique_e)<self.n_ent:

            missing_e = torch.tensor(list(set(unique_e.tolist()) - set(nonmissing_e.tolist())))
            for e in missing_e:
                e = missing_e[1]
                sub_mask = ((self.head_idx == e)|(self.tail_idx == e)).nonzero(as_tuple = False)[:,0]

                rand = torch.randperm(len(sub_mask))
                sizes = self.get_size(mask_train.shape[0],
                                 share = share,
                                 validation = False)
                mask_train[sub_mask[:sizes[0]]] = True
                if validation:
                    mask_val[sub_mask[:sizes[0]]] = False
        
        if validation:
            assert not (mask_train & mask_val).any().item()

            return mask_train, mask_val, ~(mask_train|mask_val)
        
        else:
            return mask_train, ~mask_train

    def evaluate_dicts(self):
       for i in range(self.n_facts):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].add(self.head_idx[i].item())
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].add(self.tail_idx[i].item())
            self.dict_of_rels[(self.head_idx[i].item(),
                               self.tail_idx[i].item())].add(self.relations[i].item())
    
    @staticmethod
    def get_size(n_item, share, validation = False):
        if n_item == 1:
            if validation :
                return 1, 0, 0
            else:
                return 1, 0
        elif n_item == 2:
            if validation:
                return 1, 1, 0 
            else:
                return 1, 1 
        else:
            n_train = int(n_item*share)
            assert n_train <= n_item

            if n_train == 0:
                n_train = 1 

            if not validation:
                return n_train, n_item - n_train
            else:
                if (n_item - n_train) == 1:
                    n_train -= 1
                    return n_train, 1, 1
                
                else:
                    n_val = int(int(n_item - n_train)/2)
                    return n_train, n_val, (n_item - n_train - n_val)
    
    def get_element(self,input_data,elementtype = None):
        
        if elementtype:
            if elementtype in ['from','to']:
                elements = list(set(input_data['from'].unique()).union(set(input_data['to'].unique())))
                return {e:i for i,e in enumerate(sorted(elements))}
            elif elementtype == 'rel':
                elements = list(set(input_data['rel'].unique()))
                return {e:i for i,e in enumerate(sorted(elements))}


if __name__ == '__main__':
    
    df = pd.read_csv('./dataset/FB15k/freebase_mtr100_mte100-test.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    
    KG_data = KG_Dataset(input_data = df)
    KG_train, KG_val, KG_test = KG_data.split_KG(share=0.5,validation=True)

    len(KG_train), len(KG_val), len(KG_test)

    len(KG_train)+len(KG_val)+len(KG_test) == len(KG_data)
                
