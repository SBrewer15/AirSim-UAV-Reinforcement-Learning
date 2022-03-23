
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/DQN.ipynb
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from nb_files.nb_NeuralNetwork import ModelMaker

class DQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, dropout):
        super(DQN, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.dropout = dropout
        self.model = ModelMaker(arch='tf_mobilenetv3_small_075',
                                input_channels=input_dims[0],
                                num_outputs=n_actions, dropout=self.dropout)


        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        actions = self.model(state)


        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def load_previous_checkpoint(self, previous_checkpoint):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(previous_checkpoint))
        print('... Saving as new name ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def save_weights_On_EpisodeNo(self, episode):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), f'{self.checkpoint_file}_epsiode_{episode}')