#!/usr/bin/env python3
#-*-coding:utf-8-*-

import sys
sys.path.insert(0, '.')

from dataset import *

## Dataset Donwload

baseUrl    = 'http://commondatastorage.googleapis.com/books1000/'
Downloader(
  'http://commondatastorage.googleapis.com/books1000/',
  'dataset'
).get([
  'notMNIST_large.tar.gz',
  'notMNIST_small.tar.gz'
])

## Dataset files to binary picle format

image_size = 28
dataset_small = Dataset('dataset/notMNIST_small', image_size)
dataset_large = Dataset('dataset/notMNIST_large', image_size)
dataset_small.generate_pickle()
dataset_large.generate_pickle()

# Exports merged database with all the images combined
dataset_small.export_pickle_merged(
  "dataset/test.pickle",
  train_size=10000
)

dataset_large.export_pickle_merged(
  "dataset/train.pickle",
  train_size=400000,
  valid_size=100000
)

## Loading Dataset : Training validation and test

db = Database("dataset/train.pickle", "dataset/test.pickle")

print(db.train("x").shape)
