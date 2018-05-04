# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:31:04 2018

@author: amitangshu
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import dataset
from chainer.dataset import dataset_mixin
from chainer import Link, Chain, ChainList
from chainer.cuda import to_cpu, to_gpu
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainercv.utils import read_image
from chainer.dataset import concat_examples
import scipy.io as spio
import numpy as np
import cupy as cp
from os import listdir
from os.path import isfile, join
import cv2

np.set_printoptions(threshold=np.nan)  

gpu = 1
gpu =0
chainer.cuda.get_device_from_id(gpu).use()     


mypath1='images-negative'
onlyfiles1 = [ f for f in listdir(mypath1) if isfile(join(mypath1,f)) ]
imagesnoncube = np.empty(len(onlyfiles1), dtype=object)
for n in range(0, len(onlyfiles1)):
  imagesnoncube[n] = read_image( join(mypath1,onlyfiles1[n]) )
print ('import success:', imagesnoncube.shape)

mypath2='images-positive/'
onlyfiles2 = [ f for f in listdir(mypath2) if isfile(join(mypath2,f)) ]
imagescube = np.empty(len(onlyfiles2), dtype=object)
for m in range(0, len(onlyfiles2)):
  imagescube[m] = read_image( join(mypath2,onlyfiles2[m]) )
print ('import success:', imagescube.shape)

mypath3='images_test_negative/'
onlyfiles3 = [ f for f in listdir(mypath3) if isfile(join(mypath3,f)) ]
test_imagesnoncube = np.empty(len(onlyfiles3), dtype=object)
for n in range(0, len(onlyfiles3)):
  test_imagesnoncube[n] = read_image( join(mypath3,onlyfiles3[n]) )
print ('noncube test import success:', test_imagesnoncube.shape)

mypath4='images_test_positive/'
onlyfiles4 = [ f for f in listdir(mypath4) if isfile(join(mypath4,f)) ]
test_imagescube = np.empty(len(onlyfiles4), dtype=object)
for n in range(0, len(onlyfiles4)):
  test_imagescube[n] = read_image( join(mypath4,onlyfiles4[n]) )
print ('cube test import success:', test_imagescube.shape)

labelnoncube=np.zeros((2793,1))
labelcube=np.ones((4229,1))
label1=np.vstack((labelcube, labelnoncube))
#print('label cube shape: ',labelcube.shape)
#print('label non-cube shape ',labelnoncube.shape)
imagescube = np.expand_dims(imagescube, axis=1)
imagesnoncube = np.expand_dims(imagesnoncube, axis=1)
total_images=np.vstack((imagescube, imagesnoncube))

#print('img cube shape ',imagescube.shape)
#print('img non-cube shape:',imagesnoncube.shape)
#print('total_label shape:  ',label1.shape)
#print('total_images shape:', total_images.shape)
train = np.stack((total_images,label1),axis=1)
train = np.squeeze(train,axis=2)
#print('train shape:',train.shape)
train = tuple(train)
print('Length of train data',len(train))
#print('Train[0] shape:',train[0].shape)
#print('Train[0][1]:',train[0][1])

test_labelnoncube=np.zeros((501,1))
test_labelcube=np.ones((500,1))
test_label1=np.vstack((test_labelcube, test_labelnoncube))
test_imagescube = np.expand_dims(test_imagescube, axis=1)
test_imagesnoncube = np.expand_dims(test_imagesnoncube, axis=1)
test_total_images=np.vstack((test_imagescube, test_imagesnoncube))
test = np.stack((test_total_images, test_label1),axis=1)
test = np.squeeze(test, axis=2)
test = tuple(test)
print('Length of test data', len(test))
#print('Test [0][0]', test[0][0])
#print('Test [0][1]', test[0][1])


# Choose the minibatch size.
batchsize =32

train_iter = iterators.SerialIterator(train, batchsize, repeat=True, shuffle = True)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)


class SawyerNET(Chain):
    def __init__(self):
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
            self.fc5 = L.Linear(512, 2)

    def __call__(self, x):
        h = F.copy(x, gpu)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.bn1(self.conv2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.bn2(self.conv3(h)))
        h = F.relu(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.sigmoid(self.fc5(h))

    
model=SawyerNET()

model.to_gpu(gpu)

#optimizer = optimizers.SGD()
optimizer = optimizers.Adam()
optimizer.setup(model)

max_epoch = 50

while train_iter.epoch < max_epoch:

    # ---------- One iteration of the training loop ----------
    train_batch = train_iter.next()
    #print (train_batch[0].shape)
    #print (train_batch[0][0].shape)
    #print (train_batch[0][1])
    image_train = []
    target_train = []
    for i in range (len(train_batch)):
        image_train.append(train_batch[i][0])
        target_train.append(train_batch[i][1])

    image_train = cp.asarray(image_train,  dtype=cp.float32)
    target_train = cp.asarray(target_train, dtype =cp.int8)
    target_train = F.copy(target_train, gpu)

    # Calculate the prediction of the network
    prediction_train = model(image_train)

    # Calculate the loss with softmax_cross_entropy
    loss = F.softmax_cross_entropy(prediction_train, target_train)
    #print('Batch Loss: ', loss)
    # Calculate the gradients in the network
    model.cleargrads()
    loss.backward()

    # Update all the trainable paremters
    optimizer.update()

    if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

        # Display the training loss
        print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')
        print('')

        test_losses = []
        test_accuracies = []
        while True:
            test_batch = test_iter.next()
            #print(len(test_batch))
            #print(test_batch[1][0].shape)
            #print(test_batch[1][1])

            #we can ignore the gpu_id if we are running locally
            #image_test, target_test = concat_examples(test_batch)

            
            image_test = []
            target_test = []
            
            for j in range (len(test_batch)):
                image_test.append(test_batch[j][0])
                target_test.append(test_batch[j][1])
                
            image_test = cp.asarray(image_test, dtype=cp.float32)
            target_test = cp.asarray(target_test, dtype =cp.int8)
            target_test = F.copy(target_test, gpu)
                         

            # Forward the test data
            prediction_test = model(image_test)

            # Calculate the loss
            loss_test = F.softmax_cross_entropy(prediction_test, target_test)
            test_losses.append(to_cpu(loss_test.data))

            # Calculate the accuracy
            accuracy = F.accuracy(prediction_test, target_test)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.data)

            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(test_losses), np.mean(test_accuracies)))
       # if ((np.mean(test_accuracies ))>0.9):
           # serializers.save_npz('SawyerCNN_5', model)
           # exit()

#loading the saved parameters   
print('Saving Model')     
serializers.save_npz('SawyerCNN_5', model)

# Get a test image and label
#x, t = test[i]
#plt.imshow()
#plt.savefig()
#print('label:', t)

#print(x.shape, end=' -> ')
#x = x[None, ...]
#print(x.shape)

# forward calculation of the model by sending X
#y = model(x)

# The result is given as Variable, then we can take a look at the contents by the attribute, .data.
#y = y.data

# Look up the most probable digit number using argmax
#pred_label = y.argmax(axis=1)

#print('predicted label:', pred_label[0])