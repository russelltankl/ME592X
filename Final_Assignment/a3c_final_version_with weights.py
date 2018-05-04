from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse
import os

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'

import chainer
import chainerrl
import ctypes
from chainer import functions as F
from chainer import links as L
import gym
#gym.undo_logger_setup()
import gym.wrappers
import numpy as np
from sawyer_env import Sawyer_env
from random_agent import RandomAgent


from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function


def phi(obs):
    return obs.astype(np.float32)


# class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
#     """An example of A3C feedforward softmax policy."""

#     def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
#         self.pi = policies.SoftmaxPolicy(
#             model=links.MLP(ndim_obs, n_actions, hidden_sizes))
#         self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
#         super().__init__(self.pi, self.v)

#     def pi_and_v(self, state):
#         return self.pi(state), self.v(state)


# class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
#     """An example of A3C feedforward mellowmax policy."""

#     def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
#         self.pi = policies.MellowmaxPolicy(
#             model=links.MLP(ndim_obs, n_actions, hidden_sizes))
#         self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
#         super().__init__(self.pi, self.v)

#     def pi_and_v(self, state):
#         return self.pi(state), self.v(state)


# class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
#     """An example of A3C recurrent Gaussian policy."""

#     def __init__(self, obs_size, action_size, hidden_size=200, lstm_size=128):
#         self.pi_head = L.Linear(obs_size, hidden_size)
#         self.v_head = L.Linear(obs_size, hidden_size)
#         self.pi_lstm = L.LSTM(hidden_size, lstm_size)
#         self.v_lstm = L.LSTM(hidden_size, lstm_size)
#         self.pi = policies.LinearGaussianPolicyWithDiagonalCovariance(
#             lstm_size, action_size)
#         self.v = v_function.FCVFunction(lstm_size)
#         super().__init__(self.pi_head, self.v_head,
#                          self.pi_lstm, self.v_lstm, self.pi, self.v)

#     def pi_and_v(self, state):

#         def forward(head, lstm, tail):
#             h = F.relu(head(state))
#             h = lstm(h)
#             return tail(h)

#         pout = forward(self.pi_head, self.pi_lstm, self.pi)
#         vout = forward(self.v_head, self.v_lstm, self.v)

#         return pout, vout

#gpu=0
#chainer.cuda.get_device_from_id(gpu).use()
#policy generator CNN
class SawyerNET_P(chainer.Chain,a3c.A3CModel):
    def __init__(self,obs):
        super(SawyerNET_P, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=3, out_channels=96, ksize=11, stride=2)
            self.conv2 = L.Convolution2D(in_channels=96, out_channels=256, ksize=11, stride=2)
            #I have put 192 which is the number of output channels here, but not sure what should be here exactly. 
            self.bn1= L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(in_channels=256, out_channels=1024, ksize=3, stride=2)
            self.bn2=L.BatchNormalization(1024)
            self.fc4 = L.Linear(None, 512)
            self.fc5 = L.Linear(512, 8)
           #  self.conv1 = L.Convolution2D(None, 32, ksize=3, stride=2, pad = 1)
           #  self.conv2 = L.Convolution2D(None, 64, ksize=3, stride=1, pad = 1)
           #  self.bn1= L.BatchNormalization(64)
           # # self.conv3 = L.Convolution2D(None, 64 , ksize=2, stride=2)
           #  self.bn2 = L.BatchNormalization(64)
           #  self.fc1 = L.Linear(None, 32)
           #  self.fc2 = L.Linear(32, 9)

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
        #h = F.sigmoid(self.fc5(h))
        h = self.fc5(h)
        pi = F.sigmoid((h))
        return pi
        # return h

        # #h = F.copy(x, gpu)                         #copy to device
        # h = F.relu(self.conv1(x))
        # #h = F.max_pooling_2d(h, ksize = 2, stride = 2)
        # h = F.relu(self.bn1(self.conv2(h)))
        # #print(h.shape)
        # h = F.max_pooling_2d(h,ksize= 2, stride = 1)
        # h = F.dropout(h, ratio=0.2)
        # #h = F.relu(self.bn2(self.conv3(h)))
        # h = F.relu(self.bn2((h)))
        # h = F.relu(self.fc1(h))
        # h = self.fc2(h)
        #return chainerrl.action_value.DiscreteActionValue(self.l2(h))
        #return chainerrl.action_value.DiscreteActionValue(h)
        #p = F.sigmoid((h))
        #p = p.data
        #pi = np.argmax(p)
        # print(pi)
        #print(pi.shape)
        #H1 = chainerrl.action_value.DiscreteActionValue(self.l2(h))
        
       
#v function generator cnn
class SawyerNET_V(chainer.Chain,a3c.A3CModel):
    def __init__(self,obs):
        super(SawyerNET_V, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=3, out_channels=96, ksize=11, stride=2)
            self.conv2 = L.Convolution2D(in_channels=96, out_channels=256, ksize=11, stride=2)
            #I have put 192 which is the number of output channels here, but not sure what should be here exactly. 
            self.bn1= L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(
                in_channels=256, out_channels=1024, ksize=3, stride=2)
            self.bn2=L.BatchNormalization(1024)
            self.fc4 = L.Linear(None, 512)
            self.fc5 = L.Linear(512, 1)

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
        #h = F.sigmoid(self.fc5(h))
        h = self.fc5(h)
        return h

# # class SawyerNET_V(chainer.Chain,a3c.A3CModel):
#     def __init__(self,obs):
#         super(SawyerNET_V, self).__init__()
#         with self.init_scope():
#             self.conv1 = L.Convolution2D(None, 32, ksize=3, stride=2, pad = 1)
#             self.conv2 = L.Convolution2D(None, 64, ksize=3, stride=1, pad = 1)
#             self.bn1= L.BatchNormalization(64)
#            # self.conv3 = L.Convolution2D(None, 64 , ksize=2, stride=2)
#             self.bn2 = L.BatchNormalization(64)
#             self.fc1 = L.Linear(None, 32)
#             self.fc2 = L.Linear(32, 1)

#     def __call__(self, x):                                 
#         #h = F.copy(x, gpu)                         #copy to device
#         h = F.relu(self.conv1(x))
#         #h = F.max_pooling_2d(h, ksize = 2, stride = 2)
#         h = F.relu(self.bn1(self.conv2(h)))
#         #print(h.shape)
#         h = F.max_pooling_2d(h,ksize= 2, stride = 1)
#         h = F.dropout(h, ratio=0.2)
#         #h = F.relu(self.bn2(self.conv3(h)))
#         h = F.relu(self.bn2((h)))
#         h = F.relu(self.fc1(h))
#         h = self.fc2(h)
#         return h
        #return chainerrl.action_value.DiscreteActionValue(self.l2(h))
        #return chainerrl.action_value.DiscreteActionValue(h)
       

#a3c model
class A3CSawyerNET(chainer.ChainList, a3c.A3CModel):
    #"""An example of A3C feedforward softmax policy."""

    def __init__(self, obs):
        self.pi1 = SawyerNET_P(obs)
        self.pi1 = load_pretrain('SawyerCNN_2', self.pi1)
        self.pi = policies.SoftmaxPolicy(self.pi1)
        #self.pi.to_gpu(gpu)
        self.v = SawyerNET_V(obs)
        self.v = load_pretrain('SawyerCNN_2', self.v)
        #self.v.to_gpu(gpu)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        #print(state.shape)
        return self.pi(state), self.v(state)

#model=SawyerNET()
#model.to_gpu(gpu)

#loading weights
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

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    #parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--env', type=str, default='Sawyer_env')
    #parser.add_argument('--arch', type=str, default='FFSoftmax',
                        #choices=('FFSoftmax', 'FFMellowmax', 'LSTMGaussian'))
    parser.add_argument('--arch', type=str, default='Sawyer_NET')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    #parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    #parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    #misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
    #def make_env(test):
        #env = gym.make(args.env)
        env = Sawyer_env()
        # Use different random seeds for train and test envs
        #process_seed = int(process_seeds[process_idx])
        #env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        #env.seed(env_seed)
        #if args.monitor and process_idx == 0:
            #env = gym.wrappers.Monitor(env, args.outdir)
        # Scale rewards observed by agents
        #if not test:
            #misc.env_modifiers.make_reward_filtered(
                #env, lambda x: x * args.reward_scale_factor)
        #if args.render and process_idx == 0 and not test:
            #misc.env_modifiers.make_rendered(env)
        return env

    #sample_env = gym.make(args.env)
    sample_env = Sawyer_env()
    #timestep_limit = sample_env.spec.tags.get(
        #'wrapper_config.TimeLimit.max_episode_steps')
    #obs_space = sample_env.observation_space
    obs_space = sample_env.observation
    obs_size = (obs_space).size
    #print(obs_size)
    action_space = sample_env.action_space
    n_actions = (np.asarray(sample_env.action_space)).shape[0]
    #action_space = sample_env.action_space

    # Switch policy types accordingly to action space types
    #if args.arch == 'LSTMGaussian':
        #model = A3CLSTMGaussian(obs_space.low.size, action_space.low.size)
        #model = A3CLSTMGaussian(obs_size, n_actions)
    #elif args.arch == 'FFSoftmax':
        #model = A3CFFSoftmax(obs_space.low.size, action_space.n)
        #model = A3CFFSoftmax(obs_size, n_actions)
    #elif args.arch == 'FFMellowmax':
        #model = A3CFFMellowmax(obs_space.low.size, action_space.n)
        #model = A3CFFMellowmax(obs_size, n_actions)

    model = A3CSawyerNET(obs_space)
    #model.to_gpu(gpu)
    
    #optimizer
    opt = rmsprop_async.RMSpropAsync(lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))

    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = a3c.A3C(model, opt, t_max=args.t_max, gamma=0.99,beta=args.beta, phi=phi)
    #agent = a3c.A3CModel()

    if args.load:
        agent.load(args.load)
    
    #training the agent
    n_episodes = 100
    max_episode_len = 10
    for i in range(1,n_episodes+1):
        obs = sample_env.reset()
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
            obs, reward, done = sample_env.step(action)
            #print(obs[1,1,1])
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
        obs = sample_env.reset()
        done = False
        R = 0
        t = 0
        while not done and t < 200:
            #env.render()
            action = agent.act(obs)
            obs, r, done= sample_env.step(action)
            R += r
            t += 1
        print('test episode:', i, 'R:', R)
        agent.stop_episode()

    #if args.demo:
        #env = make_env(0, True)
        #env = make_env(True)
        #eval_stats = experiments.eval_performance(
            #env=env,
            #agent=agent,
            #n_runs=args.eval_n_runs,
            #max_episode_len=10)
            #max_episode_len=timestep_limit)
        #print('n_runs: {} mean: {} median: {} stdev {}'.format(
            #args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            #eval_stats['stdev']))
    #else:
        #experiments.train_agent_async(
            #agent=agent,
            #outdir=args.outdir,
            #processes=args.processes,
            #make_env=make_env,
            #profile=args.profile,
            #steps=args.steps,
            #eval_n_runs=args.eval_n_runs,
            #eval_interval=args.eval_interval,
            #max_episode_len=10)
            #max_episode_len=timestep_limit)



if __name__ == '__main__':
    main()

