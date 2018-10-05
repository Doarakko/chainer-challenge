#!/usr/bin/env python
# -*- coding: utf-8 -*-
import _pickle as pickle
from chainer.links.caffe import CaffeFunction

LOAD_PATH = "model/VGG_FACE.caffemodel"
SAVE_PATH = "model/vgg_face.pkl" 

#Caffeモデルをpickleモデルに変換する関数
def convert_caffe_model(load_path, save_path):
    #Caffeモデルを読み込み
    caffe_model = CaffeFunction(load_path)
    #pickleモデルに変換して保存
    pickle.dump(caffe_model, open(save_path, 'wb'))
    #デバック
    print("[Save] {0}".format(save_path))

if __name__ == '__main__':
    convert_caffe_model(load_path=LOAD_PATH, save_path=SAVE_PATH)