import os 
import json 
import copy 

class ModelConfig(object):
    
    def __init__(self, **kwargs):
        
        self.model_name = kwargs.pop("model_name", None)        
        self.emb_dim = kwargs.pop("emb_dim", 100)
        self.distance_type = kwargs.pop("distance_type", 'L2')

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def get_config(cls, model_name_path):
        files = os.listdir(model_name_path)
        get_json_path = [f for f in files if f.endswith('.json')]
        
        if (len(get_json_path) == 0) or (len(get_json_path) > 1):
            raise ImportError('Please check the file of model_cofig and only one json file')

        config_dict = cls._dict_from_json_file(os.path.join(model_name_path,get_json_path[0]))
        
        if 'model_name' not in config_dict:
            raise ValueError('Please check the config of model_name')

        return cls(**config_dict)
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    
if __name__ == '__main__':
    model_path = './TransE_Model'
    config = ModelConfig.get_config(model_path)

    config