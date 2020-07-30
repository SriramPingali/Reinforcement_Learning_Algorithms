# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

import torch
from ddpg import DDPGAgent
from utilities import soft_update, transpose_to_tensor, transpose_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, num_agents, layer_params, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(layer_params['in_actor'], layer_params['hidden_in_actor'], layer_params['hidden_out_actor'],                 layer_params['out_actor'],layer_params['in_critic'], layer_params['hidden_in_critic'], layer_params['hidden_out_critic'],                   layer_params['lr_actor'], layer_params['lr_critic']) for i in range(num_agents)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        obs_all_agents = obs_all_agents.to(device)
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    
    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()

    def update(self, samples, agent_number):#, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
#         obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
#         obs, action, reward, next_obs, done = map(transpose_to_tensor, samples)
        obs_full, obs, action, reward, next_obs_full, next_obs, done = samples
#         print(obs_full.shape, obs.shape, action.shape, reward.shape, next_obs_full.shape, next_obs.shape, done.shape)
#         obs, obs_full, action, reward, next_obs, next_obs_full, done = torch.from_numpy(obs), 
#         obs_full = torch.stack(obs_full)
#         obs_full = torch.stack(obs)
#         next_obs_full = torch.stack(next_obs_full)
#         next_obs_full = torch.stack(next_obs)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs.permute(1, 0, 2))
        target_actions = torch.cat(target_actions, dim=1)
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:, agent_number].view(-1, 1))
#         action = torch.cat(action.permute(1, 0, 2), dim=1)
        action = action.reshape(-1, 4)
        critic_input = torch.cat((obs_full, action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs.permute(1, 0, 2)) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()
        

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
     
    def save_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor.state_dict(), 'checkpoint_actor_local_' + str(agent_id) + '.pth')
            torch.save(agent.critic.state_dict(), 'checkpoint_critic_local_' + str(agent_id) + '.pth')

    def load_maddpg(self):
        for agent_id, agent in enumerate(self.maddpg_agent):
            #Since the model is trained on gpu, need to load all gpu tensors to cpu:
            agent.actor.load_state_dict(torch.load('checkpoint_actor_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))
            agent.critic.load_state_dict(torch.load('checkpoint_critic_local_' + str(agent_id) + '.pth', map_location=lambda storage, loc: storage))

            agent.noise_scale = NOISE_END #initialize to the final epsilon value upon training
            
            
            




