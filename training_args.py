import copy 
import configparser
from ast import literal_eval
from dataclasses import asdict, dataclass, field

@dataclass
class TrainingArguments:
    learning_rate:float =  field(default=0.0004, metadata={"help": "learning_rate"})
    n_epochs:int =  field(default=1000, metadata={"help": "n_epochs"})
    batch_size:int =  field(default=32768, metadata={"help": "batch_size"})

    margin:float =  field(default=0.5, metadata={"help": "margin"})
    sampling_type:str =  field(default='bert', metadata={"help": "sampling_type"})
    use_cuda:float =  field(default='all', metadata={"help": "use_cuda"})

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def _dict_from_configparser(cls, args):
        
        args_dict = {}
        for section in args.sections():
            for key, val in args.items(section):
                try :
                    args_dict[key] = literal_eval(val)
                except:
                    args_dict[key] = val
        
        return args_dict
    
    @classmethod
    def get_config(cls, model_name_path):
        
        args = configparser.ConfigParser()
        args.read(model_name_path)
        
        args_dict = cls._dict_from_configparser(args)

        return cls(**args_dict)
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

if __name__ == '__main__':
    args = TrainingArguments.get_config('./training_arg.conf')