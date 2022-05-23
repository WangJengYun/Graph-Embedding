import pandas as pd

import torch 
from torch.optim import Adam

from data_format import KG_Dataset
from model import ComplEx
from Trainer import Trainer
from evaluation import LinkPredictionEvaluator

from utils.configuration_model import ModelConfig
from utils.training_args import TrainingArguments
from utils.trainer_callback import TrainiingState, MLflowCallback

def main():
    
    df = pd.read_csv('./dataset/FB15k/fb15k_dataset.csv')
    KG_data = KG_Dataset(input_data = df)
    kg_train, KG_val, KG_test = KG_data.split_KG(sizes = (483142, 50000, 59071),validation = True)
    
    config = ModelConfig.get_config('./model_files/ComplEx')
    model = ComplEx(config,
                   n_entities = kg_train.n_ent, 
                   n_relations = kg_train.n_rel)
 
    args = TrainingArguments.get_config('./training_arg.conf')
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
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
 
    trainercallback.on_train_end(args, state, eval_metrics, trainer.model)

if __name__ == "__main__":
    main()
