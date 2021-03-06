import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, mode):
        super(VAE, self).__init__()

        self.mode = mode

        if mode == 'Simple':
            self.fc1 = nn.Linear(96*96, 400)
            self.fc21 = nn.Linear(400, 20)
            self.fc22 = nn.Linear(400, 20)
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 96*96)
        elif mode == 'Complex':
            self.fc1 = nn.Linear(96*96, 10000)
            self.fc1a = nn.Linear(10000, 4096)
            self.fc11 = nn.Linear(4096, 2048)
            self.fc12 = nn.Linear(2048, 512)

            self.fc21 = nn.Linear(512, 64)
            self.fc22 = nn.Linear(512, 64)

            self.fc3 = nn.Linear(64, 512)

            self.fc31 = nn.Linear(512, 2048)
            self.fc32 = nn.Linear(2048, 4096)
            self.fc3a = nn.Linear(4096, 10000)
            self.fc4 = nn.Linear(10000, 96*96)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        # h1 = h1.view(h1.size(0), -1)

        if self.mode == 'Complex':
            h1 = F.relu(self.fc1a(h1))
            h1 = F.relu(self.fc11(h1))
            h1 = F.relu(self.fc12(h1))

        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        
        if self.mode == 'Complex':
            h3 = F.relu(self.fc31(h3))
            h3 = F.relu(self.fc32(h3))
            h3 = F.relu(self.fc3a(h3))

        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), z, mu, logvar

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # action mean range -1 to 1
        self.actor =  nn.Sequential(

                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
        )

        self.fc_actor = nn.Sequential(
                nn.Linear(64*7*7, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim),
                nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(

                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
        )

        self.fc_critic = nn.Sequential(
                nn.Linear(64*7*7, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
        )

    def forward(self, x):
        raise NotImplementedError
        '''
        x, t = x
        if t == 'actor':
            x = self.actor(x)
            x = x.view(x.size(0), -1)
            x = self.fc_actor(x)
        elif t == 'critic':
            x = self.critic(x)
            x = x.view(x.size(0), -1)
            x = self.fc_critic(x)
        return x
        '''