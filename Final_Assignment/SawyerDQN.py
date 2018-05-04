#!/usr/bin/env python

from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from chainerrl import agents
from chainerrl.agents.dqn import DQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer
import gym
import numpy as np
import rospy
rospy.init_node('image_listener', anonymous=True)
import random
import rospy
import intera_interface
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import numpy as numpy
limb = intera_interface.Limb('right')
head = intera_interface.Head()
angles = limb.joint_angles()
Dangle = limb.joint_angles()
torques = limb.joint_efforts()
velocities = limb.joint_velocities()
import time

from os import listdir
from os.path import isfile, join
limb.set_joint_position_speed(0.6)


class Sawyer_env(object):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		print('Creating Environment')

		self.path = '/home/scsrobot/ros_ws/bagfiles'
		self.image_topic = "/io/internal_camera/head_camera/image_raw"

		self.num_obs = 1
		self.num_state = 2
		self.action_space = [0,1,2,3,4,5,6,7]
		self.observation = cv2.imread('/home/scsrobot/ros_ws/bagfiles/0obs.jpg', 1)
		self.suc_counter = 0
		
		self.reward = 0
		self.sum_rewards = 0
		self.goal = [0.026978515625 ,1.5837451171875] # joint angle needed to reach cube
		
		self.acts = []
		self.state = np.asarray([angles['right_j0'], angles['right_j3']])
		print('Current State', self.state)
		self.prev_state = self.state
		self.prev_dist = np.sqrt(np.sum(np.square(np.asarray(self.state) - np.asarray(self.goal))))
		self.done = False


	def reset(self):
		print('--------------------Resetting Environment-----------------------')
		limb.set_joint_position_speed(0.3)
		num = self.num_obs
		states = angles
		head_angle= -0.1809345703125

		self.suc_counter = 0
		#initialize each new episode to random states
		x = np.random.uniform(-0.3, 0.3)
		y = np.random.uniform(-2.13, 2.13)

		angles['right_j0']= x 
		angles['right_j1']= 0.0
		angles['right_j2']= -3.142/2
		angles['right_j3']= y 
		angles['right_j4']= 3.142/2
		angles['right_j5']= 3.142/2
		angles['right_j6']= 0.0

		self.state = np.asarray([x, y])
		print('Current State', self.state)
		limb.move_to_joint_positions(angles)
		head.set_pan((-(angles['right_j0']) + head_angle), speed=1, timeout=0)
		rospy.sleep(2)

		self.done = False

		self.prev_dist = np.sqrt(np.sum(np.square(np.asarray(self.state) - np.asarray(self.goal))))
		return self.observation

	def step(self, action, fn_counter):# --> return: observation, reward, done, info
		
		rospy.init_node('image_listener', anonymous=True)

		theta = 0.1 # in radians
		head_angle = -0.1809345703125
		
		state = self.state
		self.prev_state = state

		if action == 0:
			self.acts.append([theta, -theta])
		elif action == 1:
			self.acts.append([theta, 0])
		elif action == 2:
			self.acts.append([theta, theta])
		elif action == 3:
			self.acts.append([-theta, theta])
		elif action == 4:
			self.acts.append([-theta, 0])
		elif action == 5:
			self.acts.append([-theta, -theta])
		elif action == 6:
			self.acts.append([0, theta])
		elif action == 7:
			self.acts.append([0, -theta])
		

		angles['right_j0'] += self.acts[-1][0]
		angles['right_j3'] += self.acts[-1][1]
		print(angles['right_j0'], angles['right_j3'])

		if angles['right_j0'] > 0.3 or angles['right_j0'] < -0.7: #defining limits of joint 0
			limb.set_joint_position_speed(0.5)
			angles['right_j0'] = 0.0
		else:
			limb.set_joint_position_speed(0.1)


		if angles['right_j3'] > 2.13 or angles['right_j3'] < -2.13: #defining limits of joint 3
			limb.set_joint_position_speed(0.5)
			limb.set_joint_position_speed(0.5)
			angles['right_j3'] = 0.0
		else:
			limb.set_joint_position_speed(0.1)


		bridge = CvBridge()
		self.fn_counter = fn_counter
		limb.move_to_joint_positions(angles)

		#turning head camera to counter joint 0 movement
		head.set_pan((-(angles['right_j0']) + head_angle), speed=1, timeout=0)

		#streaming camera topic as observation
		def image_callback(msg):
			cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
			self.observation = cv2_img
			cv2.imwrite(os.path.join(self.path, str(self.fn_counter) + 'obs.jpg'), cv2_img)

		rospy.Subscriber(self.image_topic, Image, image_callback)
		
		
		self.state = np.asarray([angles['right_j0'], angles['right_j3']])

		tolerance = 0.1 
		self.suc_counter += 1

		self.get_reward()
		if abs(self.state[0]-self.goal[0]) < tolerance :
			if abs(self.state[1] - self.goal [1]) < tolerance:
						print(self.state)
						with open('SuccessRate.txt', 'a') as g:
							 print('Steps to sucess: ', self.suc_counter ,file = g)
						print("I DID IT AND I AM COMING FOR YOU NEXT!!")
						self.done = True

		
		print ('Action', action, ' Reward', self.reward)
	   
		return self.observation, self.reward, self.done

	def get_reward(self):
		"""
		closer = +ve reward
		further = -ve reward
		stay = 0 reward

		"""
		dist = np.sqrt(np.sum(np.square(np.asarray(self.state) - np.asarray(self.goal))))

		dist_diff = self.prev_dist - dist
		self.reward = dist_diff * 10

		self.prev_dist = dist

env = Sawyer_env() #import created Sawyer environment without using gym

print('action space:', env.action_space) 
done = False

#to load pretrained CNN weights if using
def load_pretrain(fName, model):
    pretrain = np.load(fName)
    model.conv1.W.data = pretrain['conv1/W']
    model.conv1.b.data = pretrain['conv1/b']
    model.conv2.W.data = pretrain['conv2/W']
    model.conv2.b.data = pretrain['conv2/b']
    model.fc4.W.data = pretrain['fc4/W']
    model.fc4.b.data = pretrain['fc4/b']
    model.fc5.W.data = pretrain['fc5/W']
    model.fc5.b.data = pretrain['fc5/b']
    model.bn1.beta.data = pretrain['bn1/beta']
    model.bn1.gamma.data = pretrain['bn1/gamma']
    model.bn1.N = pretrain['bn1/N']
    model.bn1.avg_mean.data = pretrain['bn1/avg_mean']
    model.bn1.avg_var.data = pretrain['bn1/avg_var']
    model.bn2.beta.data = pretrain['bn2/beta']
    model.bn2.gamma.data = pretrain['bn2/gamma']
    model.bn2.N = pretrain['bn2/N']
    model.bn2.avg_mean.data = pretrain['bn2/avg_mean']
    model.bn2.avg_var.data = pretrain['bn2/avg_var']
    return model


gpu=0
chainer.cuda.get_device_from_id(gpu).use()
class SawyerNET(chainer.Chain):
	def __init__(self,obs):
		super(SawyerNET, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(None, 32, ksize=3, stride=2, pad = 1)
			self.conv2 = L.Convolution2D(None, 64, ksize=3, stride=1, pad = 1)
			self.bn1= L.BatchNormalization(64)
			self.bn2 = L.BatchNormalization(64)
			self.fc4 = L.Linear(None, 32)
			self.fc5 = L.Linear(32, 8)

	def __call__(self, x):                                 
		h = F.copy(x, gpu)                         #copy to device
		h = F.relu(self.conv1(x))
		h = F.relu(self.bn1(self.conv2(h)))
		h = F.max_pooling_2d(h,ksize= 2, stride = 1)
		h = F.dropout(h, ratio=0.2)
		h = F.relu(self.bn2((h)))
		h = F.relu(self.fc4(h))
		h = self.fc5(h)
		return chainerrl.action_value.DiscreteActionValue(h)

obs_1 = env.observation
obs_size = (env.observation).shape
print("obs size:",obs_size)
n_actions = (np.asarray(env.action_space)).shape[0]
q_func = SawyerNET(obs_1)
q_func.to_gpu(gpu)

optimizer = chainer.optimizers.RMSprop(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

#define a random to pick action for epsilon_greedy 
def random_num():
	x = random.randint(0,8)
	return x

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
	epsilon=0.35, random_action_func=random_num)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
	q_func, optimizer, replay_buffer, gamma, explorer, minibatch_size=16,
	replay_start_size=64, update_interval=1,
	target_update_interval=32, phi=phi)

#train the agent
counter = 0
fn_counter = 0
n_episodes = 1500
max_episode_len = 200
for i in range(1,n_episodes+1):
	print('------------------ Episode: ', counter, ' ------------------')
	obs = env.reset()
	reward = 0
	done = False
	R = 0  # return (sum of rewards)
	t = 0  # time step
	while not done and t < max_episode_len-1:
		#print(obs)
		#print(reward)
		action = agent.act_and_train(obs, reward)
		obs, reward, done = env.step(action,fn_counter)
		fn_counter+=1
		R += reward
		t += 1
	if i % 10 == 0:
		with open('Statistics.txt', 'a') as f:
			print('episode:', i,
				'R:', R,
				'statistics:', agent.get_statistics(), file = f)
	counter += 1
	agent.stop_episode_and_train(obs, reward, done)

agent.save('Final_agent')
print('Finished.')

'''
#testing the agent
for i in range(10):
	obs = env.reset()
	done = False
	R = 0
	t = 0
	while not done and t < 200:
		action = agent.act(obs)
		obs, r, done= env.step(action)
		R += r
		t += 1
	print('test episode:', i, 'R:', R)
	agent.stop_episode()
'''