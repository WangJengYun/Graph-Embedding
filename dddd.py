import pandas as pd
import numpy  as np 
from torch.optim import Adam
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import TransEModel
from torchkge.utils.datasets import load_fb15k
from torchkge.utils import Trainer, MarginLoss

from data_format import KG_Dataset


emb_dim = 128
lr = 0.0004
margin = 0.5
n_epochs = 2000
batch_size = 32768
# Load dataset
dtype_low_memory = {'from':np.object,
                    'rel':np.object,
                    'to':np.object}

df = pd.read_csv('./dataset/sub_graph_embedding_data_v1',dtype = dtype_low_memory)
KG_data = KG_Dataset(input_data = df)
kg_train, KG_val, KG_test = KG_data.split_KG(share = 0.8,validation = True)
# Define the model and criterion
model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel,
                    dissimilarity_type='L2')
criterion = MarginLoss(margin)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
trainer = Trainer(model, criterion, kg_train, n_epochs, batch_size,
                  optimizer=optimizer, sampling_type='bern', use_cuda='all',)
trainer.run()
evaluator = LinkPredictionEvaluator(model, KG_test)
evaluator.evaluate(200)
evaluator.print_results()


