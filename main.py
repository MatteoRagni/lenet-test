#!/usr/bin/env python3
#-*-coding:utf-8-*-

#
#   Copyright 2016 - Matteo Ragni
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
sys.path.insert(0, '.')

import tensorflow as tf

from dataset import *
from model import *

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
  "pool.padding": 'SAME',

  "learning.rate": 1e-4,

  "use.gpu": True
}
# Disabilito la gpu per il mio portatile, nel quale ho installato Tensorflow solo
# cpu. Per Titan posso usare ancle la gpu (e anzi la versione con CPU genera un errore)
if os.uname()[1] == "V3N0M":
    print(bcolors.FAIL + "Disabilito la gpu per questo sistema!" + bcolors.ENDC)
    configuration["use.gpu"] = False


# Dichiarazione del modello vero e proprio
model = Model(configuration)

#  _____         _      _
# |_   _| _ __ _(_)_ _ (_)_ _  __ _
#   | || '_/ _` | | ' \| | ' \/ _` |
#   |_||_| \__,_|_|_||_|_|_||_\__, |
#                             |___/
BATCH_SIZE = 20
summary_writer = tf.summary.FileWriter("/tmp/nn", model.graph)

with tf.Session(graph=model.graph) as session:
    merged_summary = tf.summary.merge_all()
    tf.global_variables_initializer().run()

    for i, x, y, vx, vy in db.batch(BATCH_SIZE):
        model.optimizer.run(
          feed_dict={ model.x: x, model.label: y, model.l6_dropout: 0.5 }
        )
        if i % 100 == 0:
            summary = session.run(merged_summary,
              feed_dict={ model.x: x, model.label: y, model.l6_dropout: 1.0 }
            )
            summary_writer.add_summary(summary, i)

            accuracy, correct_label, predict_label, label_prob = session.run(
              [model.accuracy, model.correct_label, model.predict_label, model.predict_prob],
              feed_dict={ model.x: x, model.label: y, model.l6_dropout: 1.0 }
            )
            # import pdb; pdb.set_trace()
            # questa funzione è in fondo a model.py
            print_training_information(BATCH_SIZE, i, accuracy, correct_label, predict_label, label_prob)

    print("Learning Ended")

    accuracy, correct_label, predict_label, label_prob = session.run(
      [model.accuracy, model.correct_label, model.predict_label, model.predict_prob],
      feed_dict={ model.x: db.test("x")[0:100,:,:,:], model.label: db.test("y")[0:100,:], model.l6_dropout: 1.0 }
    )
    print_training_information(100, 9999999, accuracy, correct_label, predict_label, label_prob)
