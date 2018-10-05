#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import pydotplus
import numpy as np

import chainer
from chainer import serializers
import chainer.links as L

import chainer_model

#チャネル数
CHANNEL_SIZE = 3 
#出力数
N_OUT = len(glob.glob('image/*'))

#コピーしたモデルをロードする関数
def load_copy_model(Model, load_path):
    serializers.load_npz(load_path, Model)
    model = L.Classifier(Model)
    #デバック
    print("[Load] {0}".format(load_path))
    return model

#特徴を抽出する関数
def extract_feature(model, image_data):
    #抽出した特徴を入れるリストを準備
    feature_data = []
    for image in image_data:
        #推論
        y = model.predictor(image[None, ...]).data[0]
        feature_data.append(y)  
    return feature_data

if __name__ == '__main__':
    #使用するモデルのインスタンスを生成
    Model = chainer_model.vgg_face(ch_size=CHANNEL_SIZE, n_out=N_OUT)
    
    #コピーしたモデルをロード
    model = load_copy_model(Model, load_path="./model/vgg_face.npz")
    
    #データセットをロード
    image_data = np.load("./data/image_data.npy")
    #特徴を抽出
    feature_data = extract_feature(model, image_data)
    
    #特徴データを保存するパス
    save_feature_data_path = "./data/feature_data.npy"
    #抽出した特徴データをファイルに保存
    np.save(save_feature_data_path, feature_data)
    #デバック
    print("[Save] {0}".format(save_feature_data_path))
