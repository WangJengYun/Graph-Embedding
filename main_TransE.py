import pandas as pd

import torch 
from torch.optim import Adam

from data_format import KG_Dataset
from model import TransE
from Trainer import Trainer
from evaluation import LinkPredictionEvaluator

from utils.configuration_model import ModelConfig
from utils.training_args import TrainingArguments
from utils.trainer_callback import TrainiingState, MLflowCallback


df = pd.read_csv('./dataset/FB15k/fb15k_dataset.csv')
KG_data = KG_Dataset(input_data = df)
kg_train, KG_val, KG_test = KG_data.split_KG(sizes = (483142, 50000, 59071),validation = True)

config = ModelConfig.get_config('./model_files/TransE')
model = TransE(config,
               n_entities = kg_train.n_ent, 
               n_relations = kg_train.n_rel,
               ent2idx = kg_train.ent2idx,
               rel2idx = kg_train.rel2idx)

args = TrainingArguments.get_config('./training_arg.conf')
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=2e-5)

state = TrainiingState()
state.project_name = 'FB15K'

trainercallback = MLflowCallback()
trainer = Trainer(kg_train = kg_train,
                  model = model,
                  optimizer = optimizer,
                  traning_args = args,
                  state = state,
                  trainercallback = trainercallback)
trainer.run()


Evaluator = LinkPredictionEvaluator(model, KG_test)
Evaluator.evaluate(batch_size = 200)
eval_metrics = Evaluator.evaluation_metrics()

trainercallback.on_train_end(args, state, eval_metrics, trainer.model.cpu())

