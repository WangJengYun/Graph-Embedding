import pickle
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import mlflow
import mlflow.pytorch

@dataclass
class TrainiingState:
    
    project_name:Optional[str] = None
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    total_flos: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None

    def save_to_json(self, json_path: str):
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

class MLflowCallback(object):
    def __init__(self):
        
        self._initialized = False
        self._mlflow = mlflow
        
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH

    def setup(self,args,state,model):

        if self._mlflow.active_run() is None:
            self._mlflow.set_tracking_uri('http://192.168.0.21:5213')
            self._mlflow.set_experiment(state.project_name)
            self._mlflow.active_run()

        combined_dict = args.to_dict()

        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config,**combined_dict}

        for name, value in list(combined_dict.items()):
            if len(str(value)) >= self._MAX_PARAM_VAL_LENGTH:
                print('del',name,value)
                del combined_dict[name]

        combined_dict_items = list(combined_dict.items())

        for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
            self._mlflow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))

        self._initialized = True

    def on_train_begin(self, args, state, model=None, **kwargs):
        if not self._initialized :
            self.setup(args,state,model)

    def on_log(self, args, state, logs, model=None, **kwargs):
        if not self._initialized :
            self.setup(args,state,model)

        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
        
        self._mlflow.log_metrics(metrics=metrics, step=state.global_step)

    def on_train_end(self,args, state, logs,model, **kwargs):
        
        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
        self._mlflow.log_metrics(metrics=metrics)
        
        result = {}
        result['ent2idx_table'] = model.ent2idx
        result['rel2idx_table'] = model.rel2idx
        result['entity_vector'] = model.ent_emb.weight.data.numpy()
        result['rel_vector'] = model.rel_emb.weight.data.numpy()
        
        with open('./tmp/result.pickle', 'wb') as f:
            pickle.dump(result, f)

        self._mlflow.log_artifact('./tmp/result.pickle')
        self._mlflow.pytorch.log_model(model, model.config.model_name)
        self._mlflow.end_run()

    def __del__(self):
        if self._mlflow.active_run is not None:
            self._mlflow.end_run()

if __name__ == "__main__":
    aa = TrainiingState()