#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, re, json
from PIL import Image
import numpy as np

from chainer import Chain, serializers
import chainer.functions as F
import chainer.links as L

import chainer_model

#画像サイズ
IMAGE_SIZE = 224
#チャネル数
CHANNEL_SIZE = 3 
#出力数
N_OUT = len(glob.glob('image/*'))

#学習済みのモデルをロードする関数
def load_Model():
    #使用するモデルのインスタンスを生成
    Model = chainer_model.vgg_face(ch_size=CHANNEL_SIZE, n_out=N_OUT)
    #学習済みのモデルをロード
    serializers.load_npz('log/model_epoch-1', Model)
    model = L.Classifier(Model)
    return model

#画像を数値データに変換する関数
def convert_image(img_path):
    #画像の名前を取得
    image_name = re.search(r'image_predict/(.+)', img_path)
    image_name = image_name.group(1)
    #デバック
    print('image: {0}'.format(image_name.ljust(30,' ')), end="  ")
    #白黒画像
    if CHANNEL_SIZE == 1:
        img = Image.open(img_path).convert('L') 
    #カラー画像
    else:
        img = Image.open(img_path).convert('RGB') 
    #画像サイズを変換
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)) 
    #画像データをnumpy配列に変換
    x = np.asarray(img, dtype=np.float32)
    #正規化
    x /= 255
    #numpy配列の形状を変換
    x = x.reshape(CHANNEL_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    return x

#推論する関数
def predict(model, x):
    #ラベルの辞書をロード
    with open("./data/label_dic.json", "r") as f: 
        label_json = json.load(f)
        label_dic = json.loads(label_json)
    #推論
    y = model.predictor(x[None, ...]).data.argmax(axis=1)[0]
    #辞書ビューオブジェクトでキーを取得
    keys_dic_view = label_dic.keys()
    #辞書ビューオブジェクトをリストに変換
    val = list(keys_dic_view)[y]
    #デバック
    print('predicted label: {0}\tvalue: {1}'.format(y, val))

if __name__ == '__main__':
    #画像のパスのリストを取得
    img_path_list = glob.glob('image_predict/*.jpg')
    #学習済みのモデルをロード
    model = load_Model()
    for img_path in img_path_list:
        #画像を数値データに変換
        x = convert_image(img_path)
        #推論
        predict(model, x)