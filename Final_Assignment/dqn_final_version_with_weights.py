import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from sawyer_env import Sawyer_env
#from random_agent import RandomAgent
import random

env = Sawyer_env() #import created Sawyer environment without using gym
#print('observation space:', env.observation_space)
print('action space:', env.action_space) # print action space as defined in sawyer env

obs = env.reset()
#print('initial observation:', obs) #print observation as defined in sawyer env

#testing the env with a random agent 
r=0
done = False
#agent = RandomAgent(env.action_space)
#action = agent.act(obs, r, done)
#obs, r, done, info = env.step(action)
#obs, r, done = env.step(action)
#Uncomment below to view the observation, reward obtained from using random agent
#print('next observation:', obs) 
#print('reward:', r)
#print('done:', done)
#print('info:', info)

#q_func as obtained from chainerrl quickstart guide
#class QFunction(chainer.Chain):

    #def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        #super().__init__()
        #with self.init_scope():
            #self.l0 = L.Linear(obs_size, n_hidden_channels)
            #self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            #self.l2 = L.Linear(n_hidden_channels, n_actions)

    #def __call__(self, x, test=False):
        #"""
        #Args:
            #x (ndarray or chainer.Variable): An observation
            #test (bool): a flag indicating whether it is in test mode
        #"""
        #h = F.tanh(self.l0(x))
        #h = F.tanh(self.l1(h))

gpu=0
chainer.cuda.get_device_from_id(gpu).use()



def load_pretrain(fName, model):
    pretrain = np.load(fName)
    model.conv1.W.data = pretrain['conv1/W']
    model.conv1.b.data = pretrain['conv1/b']
    model.conv2.W.data = pretrain['conv2/W']
    model.conv2.b.data = pretrain['conv2/b']
    model.conv3.W.data = pretrain['conv3/W']
    model.conv3.b.data = pretrain['conv3/b']
    #model.fc4.W.data = pretrain['fc4/W']
    #model.fc4.b.data = pretrain['fc4/b']
    #model.fc5.W.data = pretrain['fc5/W']
    #model.fc5.b.data = pretrain['fc5/b']
    #model.bn1.beta.data = pretrain['bn1/beta']
    #model.bn1.gamma.data = pretrain['bn1/gamma']
    #model.bn1.N = pretrain['bn1/N']
    #model.bn1.avg_mean.data = pretrain['bn1/avg_mean']
    #model.bn1.avg_var.data = pretrain['bn1/avg_var']
    #model.bn2.beta.data = pretrain['bn2/beta']
    #model.bn2.gamma.data = pretrain['bn2/gamma']
    #model.bn2.N = pretrain['bn2/N']
    # model.bn2.avg_mean.data = pretrain['bn2/avg_mean']
    # model.bn2.avg_var.data = pretrain['bn2/avg_var']
    return model

#modfied Qfunc to obtain Qvalues for every action by importing obs from env
#class SawyerNET(chainer.Chain):
    #def __init__(self,obs):
        #super(SawyerNET, self).__init__()
        #with self.init_scope():
            #self.conv1 = L.Convolution2D(None, 96, ksize=11, stride=2, pad = 1)
            #self.conv2 = L.Convolution2D(None, 256, ksize=11, stride=2, pad = 1)
            #self.bn1= L.BatchNormalization(256)
           # self.conv3 = L.Convolution2D(None, 64 , ksize=2, stride=2)
            #self.bn2 = L.BatchNormalization(1024)
            #self.fc4 = L.Linear(None, 512)
            #self.fc5 = L.Linear(512, 9)

    #def __call__(self, x):                                 
        #h = F.copy(x, gpu)                         #copy to device
        #h = F.relu(self.conv1(x))
        #h = F.max_pooling_2d(h, ksize = 2, stride = 2)
        #h = F.relu(self.bn1(self.conv2(h)))
        #print(h.shape)
        #h = F.max_pooling_2d(h,ksize= 2, stride = 1)
        #h = F.dropout(h, ratio=0.2)
        #h = F.relu(self.bn2(self.conv3(h)))
        #h = F.relu(self.bn2((h)))
        #h = F.relu(self.fc4(h))
        #h = self.fc5(h)
        #H = F.sigmoid((h))
        #return chainerrl.action_value.DiscreteActionValue(self.l2(h))
        #return chainerrl.action_value.DiscreteActionValue(h)

class SawyerNET(chainer.Chain):
    def __init__(self,obs):
        super(SawyerNET, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=3, out_channels=96, ksize=11, stride=2)
            self.conv2 = L.Convolution2D(
                in_channels=96, out_channels=256, ksize=11, stride=2)
            #I have put 192 which is the number of output channels here, but not sure what should be here exactly. 
            self.bn1= L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(
                in_channels=256, out_channels=1024, ksize=3, stride=2)
            self.bn2=L.BatchNormalization(1024)
            self.fc4 = L.Linear(None, 512)
            self.fc5 = L.Linear(512, 8)

    def __call__(self, x):
        #h = F.copy(x, gpu)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.bn1(self.conv2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        #h = F.dropout(h, ratio=0.2)
        h = F.relu(self.bn2(self.conv3(h)))
        h = F.relu(self.fc4(h))
        #if chainer.config.train:
            #return self.fc5(h)
        h = F.sigmoid(self.fc5(h))
        return chainerrl.action_value.DiscreteActionValue(h)

#model=SawyerNET()
#model.to_gpu(gpu)

#print(env.observation)
#print(env.action_space)
obs_1 = env.observation
#print(obs_1)
obs_size = (env.observation).shape
print(obs_size)
n_actions = (np.asarray(env.action_space)).shape[0]
#q_func = QFunction(obs_size, n_actions)
q_func = SawyerNET(obs_1)
q_func = load_pretrain('SawyerCNN_2', q_func)
q_func.to_gpu(gpu)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

#define a random to pick action for epsilon_greedy 
def random_num():
    x = random.randint(0,7)
    return x

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=random_num)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
#replay_buffer.to_gpu(gpu)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=50, update_interval=1,
    target_update_interval=100, phi=phi)#, gpu=gpu)
#agent.to_gpu(gpu)

#train the agent
n_episodes = 200
max_episode_len = 10
for i in range(1,n_episodes+1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len-1:
        # Uncomment to watch the behaviour
        # env.render()
        #print(obs)
        #print(reward)
        action = agent.act_and_train(obs, reward)
        obs, reward, done = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')

#testing the agent
for i in range(10):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        #env.render()
        action = agent.act(obs)
        obs, r, done= env.step(action)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    agent.stop_episode()