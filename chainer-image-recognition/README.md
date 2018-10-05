# Chainer-Image-Recognition

## Overview
ローカルの画像を使用して, 画像認識を行う.

## Description
・make_dataset.py：データセット作成  
・chainer_model.py：使用するモデル  
・train.py：学習  
・predict.py：推論  

## Directory structure
image：学習させる画像を入れるディレクトリ. image内にクラスごとにディレクトリを作成し, その中に画像を入れる.  
image_predict：推論する画像を入れるディレクトリ    
data：画像を数値データに変換したファイル, ラベルのファイル等が保存されるディレクトリ  
noise：画像を数値データに変換する際に, 失敗した画像の移動先  
log：学習した際のモデルや, accracy, lossのグラフが保存されるディレクトリ  

## Demo


## Requirement
・Anaconda3-5.0.0  
・pip 9.0.1  
・Chainer 3.1.0  

## Usage

