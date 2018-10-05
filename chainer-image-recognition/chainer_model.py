#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class my_model(Chain):
    def __init__(self, ch_size, n_out):
        super(my_model, self).__init__(
            conv1=L.Convolution2D(ch_size, 32, 3, pad=1),
            conv2=L.Convolution2D(None, 64, 3, pad=1),
            conv3=L.Convolution2D(None, 64, 3, pad=1),
            
            fc4=L.Linear(None, 512),
            fc5=L.Linear(None, n_out),
        )
        self.train = True
 
    def __call__(self, x):
        relu1 = F.relu(self.conv1(x))
        pool1 = F.max_pooling_2d(relu1, 2, stride=2)
        drop1 = F.dropout(pool1, ratio=0.25)

        #drop1 = F.flatten(drop1)
        relu2 = F.relu(self.conv2(drop1))
        
        conv3 = F.relu(self.conv3(relu2))
        pool3 = F.max_pooling_2d(conv3, 2, stride=2)
        drop3 = F.dropout(pool3, ratio=0.25)

        relu4 = F.relu(self.fc4(drop3))
        drop4 = F.dropout(relu4, ratio=0.5)

        prob = F.softmax(self.fc5(drop4))
        return prob

class vgg_face(Chain):
    insize = 224
    def __init__(self, ch_size, n_out):
        super(vgg_face, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, pad=1),
            conv2_1=L.Convolution2D(64, 128, 3, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1),
            conv3_1=L.Convolution2D(128, 256, 3, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1),
            conv4_1=L.Convolution2D(256, 512, 3, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, pad=1),
            conv5_1=L.Convolution2D(512, 512, 3, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(4096, 4096),
            my_fc8=L.Linear(4096, n_out),
        )
        self.train = True
 
    def __call__(self, x):
        relu1_1 = F.relu(self.conv1_1(x))
        relu1_2 = F.relu(self.conv1_2(relu1_1))
        pool1 = F.max_pooling_2d(relu1_2, 2, stride=2)
        relu2_1 = F.relu(self.conv2_1(pool1))
        relu2_2 = F.relu(self.conv2_2(relu2_1))
        pool2 = F.max_pooling_2d(relu2_2, 2, stride=2)
        relu3_1 = F.relu(self.conv3_1(pool2))
        relu3_2 = F.relu(self.conv3_2(relu3_1))
        relu3_3 = F.relu(self.conv3_3(relu3_2))
        pool3 = F.max_pooling_2d(relu3_3, 2, stride=2)
        relu4_1 = F.relu(self.conv4_1(pool3))
        relu4_2 = F.relu(self.conv4_2(relu4_1))
        relu4_3 = F.relu(self.conv4_3(relu4_2))
        pool4 = F.max_pooling_2d(relu4_3, 2, stride=2)
        relu5_1 = F.relu(self.conv5_1(pool4))
        relu5_2 = F.relu(self.conv5_2(relu5_1))
        relu5_3 = F.relu(self.conv5_3(relu5_2))
        pool5 = F.max_pooling_2d(relu5_3, 2, stride=2)
        relu6 = F.relu(self.fc6(pool5))
        drop6 = F.dropout(relu6, ratio=0.5)
        relu7 = F.relu(self.fc7(drop6))
        drop7 = F.dropout(relu7, ratio=0.5)
        prob = F.softmax(self.my_fc8(drop7))
        return prob

class Alex(Chain):
    def __init__(self, ch_size, n_out):
        super(Alex, self).__init__(
            conv1 = L. Convolution2D(ch_size, 96, 11, stride=4),
            conv2 = L. Convolution2D(96,256, 5, pad=2),
            conv3 = L. Convolution2D(256, 384, 3, pad=1),
            conv4 = L. Convolution2D(384, 384, 3, pad=1),
            conv5 = L. Convolution2D(384, 256, 3, pad=1),
            fc6 = L.Linear(None, 4096),
            fc7 = L.Linear(None, 4096),
            fc8 = L.Linear(None, n_out),
        )
        self.train = True
    def __call__(self, x, train=True):
        h = F.max_pooling_2d(F.relu(F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h