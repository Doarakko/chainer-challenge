#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, re
import numpy as np
import _pickle as pickle

from chainer import serializers
import chainer.links as L

import chainer_model
import copy_model as cm

#チャネル数
CHANNEL_SIZE = 3 
#出力数
N_OUT = len(glob.glob('image/*'))

#pickleモデルをロードする関数
def load_pickle_model(load_path):
    model = pickle.load(open(load_path, "rb"))
    return model

#コピーしたモデルを保存する関数
def save_copy_model(Model, load_path, save_path):
    #モデルを作成
    model = L.Classifier(Model)
    #pickleモデルをロード
    original_model = load_pickle_model(load_path)
    #学習済みのモデルのパラメータをコピー
    cm.copy_model(original_model, model.predictor)
    #モデルを保存
    serializers.save_npz(save_path, model.predictor)
    #デバック
    print("[Save] {0}".format(save_path))

if __name__ == '__main__':
    #使用するモデルのインスタンスを生成
    Model = chainer_model.vgg_face(ch_size=CHANNEL_SIZE, n_out=N_OUT)
    #コピーしたモデルを保存
    save_copy_model(Model, load_path="./model/vgg_face.pkl", save_path="./model/vgg_face.npz")