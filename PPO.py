import argparse
import os
import gym
import torch
from torch.optim import Adam
from core.util import get_class_attr_val
from config import Config
from buffer import ReplayBuffer
from model import VAE, ActorCritic
from trainer import Trainer
from tester import Tester
from torch import nn
from torch.distributions import Categorical, MultivariateNormal

class PPO:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        # self.buffer = deque(maxlen=self.config.max_buff)
        # self.buffer = ReplayBuffer(self.config.max_buff)
        self.buffer = ReplayBuffer()

        self.policy = ActorCritic(self.config.state_dim, self.config.action_dim)
        self.policy_old = ActorCritic(self.config.state_dim, self.config.action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = Adam(self.policy.parameters(), lr=self.config.learning_rate, betas=self.config.betas)

        self.MseLoss = nn.MSELoss()

        self.action_var = torch.full((self.config.action_dim,), self.config.action_std*self.config.action_std)

        if self.config.use_cuda:
            self.cuda()
    
    def act(self, state):
            
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        # if self.config.use_cuda:
        #     state.cuda()

        action_mean = self.policy_old.actor(state)
        cov_mat = torch.diag(self.action_var).cuda() #if self.config.use_cuda else torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().data.numpy().flatten()

    def evaluate(self, state, action):
        
        action_mean = self.policy.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).cuda() #if self.config.use_cuda else torch.diag_embed(action_var)
        if self.config.use_cuda:
            cov_mat.cuda()            
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.policy.critic(state)
        # state_value = self.policy((state,'critic'))

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def learning(self, fr):

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.config.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states).cuda(), 1).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions).cuda(), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs), 1).cuda().detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.config.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.config.eps_clip, 1+self.config.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())        

    def cuda(self):
        self.policy.cuda()
        self.policy_old.cuda()

    def load_weights(self, model_path):
        policy = torch.load(model_path)
        if 'policy' in policy:
            self.policy.load_state_dict(policy['policy'])
            self.policy_old.load_state_dict(policy['policy'])
        else:
            self.policy.load_state_dict(policy)
            self.policy_old.load_state_dict(policy)

    def save_model(self, output, name=''):
        torch.save(self.policy.state_dict(), '%s/policy_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_policy'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'policy': self.policy.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.policy.load_state_dict(checkpoint['policy'])
        self.policy_old.load_state_dict(checkpoint['policy'])
        return fr

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--env', default='BipedalWalker-v3', type=str, help='gym environment')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    args = parser.parse_args()

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.frames = 2000000
    config.use_cuda = True
    config.learning_rate = 0.001
    config.max_buff = 50000
    config.update_tar_interval = 4000
    # config.batch_size = 32
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 50000
    config.win_reward = 1000
    config.win_break = True

    config.max_episodes = 10000        # max training episodes
    config.max_timesteps = 1500        # max timesteps in one episode
    config.action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    config.K_epochs = 80               # update policy for K epochs
    config.eps_clip = 0.2              # clip parameter for PPO
    config.betas = (0.9, 0.999)

    env = gym.make(config.env)

    config.state_dim = 20 # env.observation_space.shape[0]

    config.action_type = 'Continuous'
    config.action_dim = env.action_space.shape[0]
    print('Continuous', config.state_dim, config.action_dim)

    agent = PPO(config)

    VAEmode = 'Simple' # Complex

    model = VAE(VAEmode)
    if torch.cuda.is_available():
        model.cuda()
    policy = torch.load('./vae_'+VAEmode+'.pth')
    model.load_state_dict(policy)

    if args.train:
        # trainer = Trainer(agent, env, config)
        trainer = Trainer(agent, model, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        # tester = Tester(agent, env, args.model_path)
        tester = Tester(agent, model, env, args.model_path)
        tester.test()

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        # trainer = Trainer(agent, env, config)
        trainer = Trainer(agent, model, env, config)
        trainer.train(fr)
