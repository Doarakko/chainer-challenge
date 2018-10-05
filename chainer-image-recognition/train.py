#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import pydotplus
import numpy as np

from chainer import Variable, optimizers, Chain, serializers, iterators, training
from chainer.training import extensions
from chainer.datasets import tuple_dataset, split_dataset_random
import chainer.functions as F
import chainer.links as L

import chainer_model

#バッチサイズ
BATCH_SIZE = 10
#エポック数
MAX_EPOCH = 10
#チャネル数
CHANNEL_SIZE = 3 
#GPUのID
GPU_ID = -1
#出力数
N_OUT = len(glob.glob('image/*'))

#データセットをロードする関数
def load_dataset():
    image_data = np.load("./data/image_data.npy")
    label_data = np.load("./data/label_data.npy")
    #numpy配列をTupleDataset型に変換
    dataset = tuple_dataset.TupleDataset(image_data, label_data)
    #学習データとテストデータに分割
    train_data, test_data = (split_dataset_random(dataset=dataset, first_size=int(len(dataset)*0.8), seed=0))
    #デバック
    print("train_data: {0}\ttest_data: {1}".format(len(train_data), len(test_data)))
    return train_data, test_data

#学習する関数
def train(Model, train_data, test_data):
    #モデルを作成
    model = L.Classifier(Model)
    
    #model.to_gpu(GPU_ID) 

    #最適化手法
    #AdaDelta
    #optimizer = optimizers.AdaDelta()
    #AdaGrad
    #optimizer = optimizers.AdaGrad()
    #Adam
    #optimizer = optimizers.Adam()
    #MomentumSGD
    #optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    #NesterovAG
    #optimizer = optimizers.NesterovAG()
    #RMSprop()
    optimizer = optimizers.RMSprop()  
    #RMSpropGraves
    #optimizer = optimizers.RMSpropGraves()
    #SGD
    #optimizer=optimizers.SGD()
    #SMORMS3
    #optimizer = optimizers.SMORMS3()
    
    optimizer.setup(model)

    #Iteratorを準備
    train_iter = iterators.SerialIterator(train_data, BATCH_SIZE)
    test_iter = iterators.SerialIterator(test_data, BATCH_SIZE, repeat=False, shuffle=True)

    #UpdaterにIteratorとOptimizerを渡す
    updater = training.StandardUpdater(train_iter, optimizer, device=GPU_ID)
    #TrainerにUpdaterとエポック数を渡す
    trainer = training.Trainer(updater, (MAX_EPOCH, 'epoch'), out='log')

    #TrainerにExtensionを追加
    #ログを自動的にファイルに保存
    trainer.extend(extensions.LogReport())
    #プログレスバーを表示
    trainer.extend(extensions.ProgressBar())
    #学習率を取得
    #trainer.extend(extensions.observe_lr())
    #標準出力
    #trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    #学習中のモデルを指定されたタイミングで評価
    trainer.extend(extensions.Evaluator(test_iter, model, device=GPU_ID))
    #lossのグラフを描画・保存
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    #accuracyのグラフを描画・保存
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    #モデルの構造をGraphvizのdot形式で保存
    trainer.extend(extensions.dump_graph('main/loss', out_name='graph.dot'))
    #Trainerオブジェクトを1エポックごとに保存
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    #model.predictorを1エポックごとに保存
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    
    #学習を開始
    trainer.run()

    #dotファイルを読み込み
    model = pydotplus.graphviz.graph_from_dot_file('log/graph.dot')
    #dotファイルをpngで保存
    model.write_png('log/graph.png')

if __name__ == '__main__':
    #データセットをロード
    train_data, test_data = load_dataset()

    #使用するモデルのインスタンスを生成
    Model = chainer_model.my_model(ch_size=CHANNEL_SIZE, n_out=N_OUT)

    #学習
    train(Model, train_data, test_data)