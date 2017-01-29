#!/usr/bin/env python3
#-*-coding:utf-8-*-

import sys
sys.path.insert(0, '.')

from dataset import *

#  ___                  _              _
# |   \ _____ __ ___ _ | |___  __ _ __| |
# | |) / _ \ V  V / ' \| / _ \/ _` / _` |
# |___/\___/\_/\_/|_||_|_\___/\__,_\__,_|

if False:
    baseUrl    = 'http://commondatastorage.googleapis.com/books1000/'
    Downloader(
    'http://commondatastorage.googleapis.com/books1000/',
    'dataset'
    ).get([
    'notMNIST_large.tar.gz',
    'notMNIST_small.tar.gz'
    ])

#  ___       _                _     ___                             _   _
# |   \ __ _| |_ __ _ ___ ___| |_  | _ \_ _ ___ _ __  __ _ _ _ __ _| |_(_)___ _ _
# | |) / _` |  _/ _` (_-</ -_)  _| |  _/ '_/ -_) '_ \/ _` | '_/ _` |  _| / _ \ ' \
# |___/\__,_|\__\__,_/__/\___|\__| |_| |_| \___| .__/\__,_|_| \__,_|\__|_\___/_||_|
#                                              |_|

if False:
    image_size = 28
    no_classes = 10
    dataset_small = Dataset('dataset/notMNIST_small', image_size, no_classes)
    dataset_large = Dataset('dataset/notMNIST_large', image_size, no_classes)
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

#  _                 _ _             ___       _                _
# | |   ___  __ _ __| (_)_ _  __ _  |   \ __ _| |_ __ _ ___ ___| |_
# | |__/ _ \/ _` / _` | | ' \/ _` | | |) / _` |  _/ _` (_-</ -_)  _|
# |____\___/\__,_\__,_|_|_||_\__, | |___/\__,_|\__\__,_/__/\___|\__|
#                            |___/

db = Database("dataset/train.pickle", "dataset/test.pickle")

print("Training size: dim(x) = {} -> dim(y) = {}".format(db.train("x").shape, db.train("y").shape))
print("Validation size: dim(x) = {} -> dim(y) = {}".format(db.valid("x").shape, db.valid("y").shape))
print("Testing size: dim(x) = {} -> dim(y) = {}".format(db.test("x").shape, db.test("y").shape))

#  __  __         _     _   ___         _               _   _
# |  \/  |___  __| |___| | |   \ ___ __| |__ _ _ _ __ _| |_(_)___ _ _
# | |\/| / _ \/ _` / -_) | | |) / -_) _| / _` | '_/ _` |  _| / _ \ ' \
# |_|  |_\___/\__,_\___|_| |___/\___\__|_\__,_|_| \__,_|\__|_\___/_||_|

configuration = {
  "image.size": 28,
  
  "patch.1": 5,
  "patch.2": 5,
  "depth.1": 1,
  "depth.2": 32,
  "depth.3": 64,
  # dopo 2 max pooling la dimensione della immagine è 7 x 7 x 64
  "fc.1.in": (7 * 7 * 64),
  "fc.1.out": 1024,
  "fc.2.out": 10,

  "conv.stride": [1,1,1,1],
  "conv.padding": 'SAME',

  "pool.size": [1,2,2,1],
  "pool.stride": [1,2,2,1],
  "pool.padding": 'SAME'
}
