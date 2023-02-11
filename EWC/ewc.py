import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader

class EWC:
    def __init__(self, model, criterion, lr=0.001, weight=1000000):
        self.model = model
        self.weight = weight
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())
    
    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            output = F.log_softmax(self.model(input), dim=1)
            log_liklihoods.append(output[:, target])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)
    
    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        if hasattr(self, "classifier"):
            list += self.classifier.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        list += self.convD.list_init_layers()
        return list
    
    def train_gen_classifier_on_stream(self, model, datastream, iters=2000, loss_cbs=list(), eval_cbs=list()):

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters + 1))

        for batch_id, (x,y,_) in enumerate(datastream, 1):

            if batch_id > iters:
                break

            # Move data to correct device
            x = x.to(model._device())
            y = y.to(model._device())

            # Cycle through all classes. For each class present, take training step on corresponding generative model
            for class_id in range(model.classes):
                if class_id in y:
                    x_to_use = x[y==class_id]
                    loss_dict = getattr(model, "vae{}".format(class_id)).train_a_batch(x_to_use)
                    # NOTE: this way, only the [lost_dict] of the last class present in the batch enters into the [loss_cb]

            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_id, loss_dict)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_id, context=None)

        # Close progres-bar(s)
        progress.close()