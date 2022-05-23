from tqdm.autonotebook import tqdm
from dataloader import DataLoader

class Trainer:
    def __init__(self,
                 kg_train,
                 model,
                 optimizer,
                 traning_args,
                 state,
                 trainercallback):

        self.kg_train = kg_train
        
        self.model = model  
        self.loss_name = self.model.config.loss_name
        self.criterion = model.loss_fn()
        self.optimizer = optimizer 

        self.state = state
        self.args = traning_args
        self.trainercallback = trainercallback
        self.trainercallback.on_train_begin(self.args, self.state, self.model)

        if self.args.use_cuda == 'cuda':
            self.model.cuda()
            self.criterion.cuda()
    
    def training_process(self, current_batch):
        self.optimizer.zero_grad()
        h, t, r = current_batch['h'], current_batch['t'], current_batch['r']
        nh, nt  = current_batch['nh'], current_batch['nt']

        p,n = self.model((h, t, r),(nh, nt, r))
        loss = self.criterion(p,n)

        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def run(self):
        
        data_loader = DataLoader(self.kg_train,neg_sampling = self.args.sampling_type, batch_size = self.args.batch_size, device = self.args.use_cuda)
        
        logs = {}      
        epochs_iterator = tqdm(range(self.args.n_epochs),unit = 'epoch')
        for epoch in epochs_iterator:
            sum_ = 0
            for i, batch  in enumerate(data_loader):
                loss = self.training_process(batch)
                sum_ += loss

            self.state.global_step = epoch
            logs[self.loss_name] = sum_ / len(data_loader)
            self.trainercallback.on_log(self.args, self.state,logs,self.model)
            
            epochs_iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, sum_ / len(data_loader)))
            
            self.model.normalize_parameters()