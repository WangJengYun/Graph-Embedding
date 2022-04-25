from Dataloader import TrainDataLoader
# from torchkge.utils.training import TrainDataLoader
from tqdm.autonotebook import tqdm

class Trainer:
    def __init__(self, 
                 model,
                 kg_train,
                 state,
                 trainercallback,
                 traning_args,
                 criterion,
                 optimizer):

        self.model = model
        self.kg_train = kg_train
        self.state = state
        self.args = traning_args
        self.trainercallback = trainercallback
        self.criterion = criterion
        self.optimizer = optimizer 

        self.n_triple = len(kg_train) 

        self.trainercallback.on_train_begin(self.args, self.state, self.model)
    
    def training_process(self, current_batch):
        self.optimizer.zero_grad()
        h, t, r = current_batch['h'], current_batch['t'], current_batch['r']
        nh, nt  = current_batch['nh'], current_batch['nt']

        p,n = self.model(h, t, r , nh, nt)
        loss = self.criterion(p,n)

        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def run(self):

        if self.args.use_cuda in ['all','batch']:
            self.model.cuda()
            self.criterion.cuda()
        
        epochs_iterator = tqdm(range(self.args.n_epochs),unit = 'epoch')

        data_loader = TrainDataLoader(self.kg_train,
                                      batch_size = self.args.batch_size,
                                      sampling_type = self.args.sampling_type,
                                      use_cuda = self.args.use_cuda)
        logs = {}
        for epoch in epochs_iterator:
            sum_ = 0
            for i, batch  in enumerate(data_loader):
                loss = self.training_process(batch)
                sum_ += loss

            self.state.global_step = epoch
            logs['loss'] = sum_ / len(data_loader)
            self.trainercallback.on_log(self.args, self.state,logs,self.model)
            
            epochs_iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, sum_ / len(data_loader)))
            self.model.normalize_parameters()

        # self.trainercallback.on_train_end(self.args, self.state, self.model)