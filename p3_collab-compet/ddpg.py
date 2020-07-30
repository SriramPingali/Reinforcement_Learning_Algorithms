# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np

from collections import deque
import random
from utilities import transpose_list

NOISE_START=1.0
NOISE_END=0.1
NOISE_REDUCTION=0.999
EPISODES_BEFORE_TRAINING = 300
NUM_LEARN_STEPS_PER_ENV_STEP = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-4, lr_critic=1.0e-4):
        super(DDPGAgent, self).__init__()
        
        self.state_size = in_actor
        self.action_size = out_actor 

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )
#         self.noise = OUNoise(action_size) #single agent only
        self.noise_scale = NOISE_START

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)


#     def act(self, obs, noise=0.0):
#         obs = obs.to(device)
#         action = self.actor(obs) + noise*self.noise.noise().to(device)
#         return action
    
    def act(self, states, i_episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        
        if self.noise_scale > NOISE_END:
            #self.noise_scale *= NOISE_REDUCTION
            self.noise_scale = NOISE_REDUCTION**(i_episode-EPISODES_BEFORE_TRAINING)
        #else keep the previous value
        
        if not add_noise:
            self.noise_scale = 0.0
                                    
#         states = torch.from_numpy(states).float().to(DEVICE)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)#.cpu().data.numpy()
        self.actor.train()
        
        #add noise
        actions += self.noise_scale*self.add_noise2() #works much better than OU Noise process
#         actions += self.noise_scale*self.noise.noise()
        
        return actions

    def add_noise2(self):
#         noise = 0.5*np.random.randn(self.action_size) #sigma of 0.5 as sigma of 1 will have alot of actions just clipped
        noise = 0.5*torch.rand(self.action_size).to(device)
        return noise
    
    def reset(self):
        self.noise.reset()

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise().to(device)
        return action
    
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float().to(device)
    
class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        
        input_to_buffer = transpose_list(transition)
    
        for item in input_to_buffer:
            self.deque.append(item)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)
