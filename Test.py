import torch 
from torch.optim import Adam
import pandas as pd
from loss import MarginLoss
from data_format import KG_Dataset
from model.TransE import TransE
from Trainer import Trainer
from Evaluation import RelationPredictionEvaluator, LinkPredictionEvaluator

from configuration_model import ModelConfig
from training_args import TrainingArguments
from trainer_callback import TrainiingState, MLflowCallback

def main():
    # Define some hyper-parameters for training
    
    df = pd.read_csv('./dataset/FB15k/fb15k_dataset.csv')
    KG_data = KG_Dataset(input_data = df)
    kg_train, KG_val, KG_test = KG_data.split_KG(sizes = (483142, 50000, 59071),validation = True)
    
    config = ModelConfig.get_config('./TransE_Model')
    model = TransE(config,
                   n_entities = kg_train.n_ent, 
                   n_relations = kg_train.n_rel)
 
    args = TrainingArguments.get_config('./training_arg.conf')
    criterion = MarginLoss(args.margin)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    state = TrainiingState()
    state.project_name = 'FB15K'
    trainercallback = MLflowCallback()
    trainer = Trainer(model = model,
                      kg_train = kg_train,
                      traning_args = args,
                      state = state,
                      trainercallback = trainercallback,
                      criterion = criterion,
                      optimizer = optimizer)
 
    trainer.run()
    
    eval_logs = {}
    Evaluator = LinkPredictionEvaluator(model, KG_test)
    Evaluator.evaluate(batch_size = 200)
    eval_metrics = Evaluator.mean_rank()
    eval_logs['rank'] = eval_metrics[0]
    eval_logs['rank_by_filter'] = eval_metrics[1]
    # saving_model_path = './model_files/model_20220410.pt'
    # torch.save(trainer.model,saving_model_path)
    # saving_model = torch.load(saving_model_path)

    trainercallback.on_train_end(args, state,eval_logs,trainer.model)

if __name__ == "__main__":
    main()
