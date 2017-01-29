#!/usr/bin/env python3
#-*-coding:utf-8-*-

#
#   Copyright 2016 - Matteo Ragni, University of Trento
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

from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
import glob
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

import pdb

class Downloader:
    def __init__(self, baseUrl, distDir):
        r"""
        Inizializza il downloader con:
         * un URL da cui scaricare il dataset in forma di tarball
         * una path dove salvare le tarball
        """
        self.baseUrl = baseUrl
        self.baseDir = distDir
        if not os.path.isdir(distDir):
            os.mkdir(distDir)

    def dwn_progress(self, count, blkSize, totSize):
        """
        Funzione che stampa a schermo l'avanzamento del download
        """
        percent = int(count * blkSize * 100 / totSize)
        if percent % 5 == 0:
            print(("\r" * 4) + ("%3d%%" % percent), end="")

    def download(self, filename, force=False):
        r"""
        Esegue il download di `filename` nella cartella impostata per il download
        alla inizializzazione mediante baseDir.

        La flag `force` forza il download quando trova un file nella directory
        con lo stesso nome
        """
        if os.path.exists(os.path.join(self.baseDir,filename)) and not force:
            print("{}: Already donwloaded. Skipping.".format(filename))
            return self
        else:
            filename, _ = urlretrieve(
              self.baseUrl + filename,
              os.path.join(self.baseDir,filename),
              reporthook=self.dwn_progress
            )
        return self

    def extract(self, filename, force=False):
        r"""
        Estrae la tarball `filename` relativamente alla baseDir.
        Se trova già una directory con lo stesso nome scaricata, skippa
        a meno che non sia settata la flag `force` a `True`
        """
        root = os.path.splitext(os.path.splitext(filename)[0])[0]
        filename = os.path.join(self.baseDir, filename)
        if os.path.isdir(os.path.join(self.baseDir, root)) and not force:
            print("{}: already extracted. Skipping.".format(filename))
        else:
            tar = tarfile.open(filename)
            tar.extractall(path=self.baseDir)
            tar.close()
        return self

    def get(self, filename, force=False):
        r"""
        Scarica ed estrae i dataset. Se il dataset è fornito come una list
        esegue le operazioni per ognuna degli elementi della lista
        """
        if type(filename) == list:
            for fn in filename:
                self.download(fn, force)
                self.extract(fn, force)
        else:
            self.download(filename, force)
            self.extract(filename, force)



class Dataset:
    def __init__(self, path, size, classes, depth=255.0):
        self.path = path
        self.size = size
        self.depth = depth
        self.classes = classes
        self.pickles = []

    def import_class(self, cls):
        #pdb.set_trace()
        files = os.listdir(os.path.join(self.path, cls))
        dataset = np.ndarray(
          shape=(len(files), self.size, self.size),
          dtype=np.float32
        )
        image_no = 0

        for image in files:
            # if image_no % 1000 == 0:
            #    print("Loading image {}: {} of {}".format(image, image_no, len(files)))
            image_file = os.path.join(self.path, cls, image)
            try:
                # # # #
                # Rascaling image
                #  Image - 127.5
                #  -------------  \in [-1, 1]
                #      255.0
                image_bin = (ndimage.imread(image_file).astype(float) - (self.depth/2.0)) / self.depth
                # # # #
                if image_bin.shape != (self.size, self.size):
                    raise Exception("{}: wrong image size: {}".format(image_bin, image_bin.shape))
                dataset[image_no, :, :] = image_bin
                image_no = image_no + 1
            except IOError as e:
                print("{}: Reading error. Skipping".format(image))

        dataset = dataset[0:image_no, :, :]
        print("Dataset {}: shape = {}, mean = {}, std = {}".format(
          cls,
          dataset.shape,
          np.mean(dataset),
          np.std(dataset)
        ))
        return dataset

    def generate_pickle(self, force=False):
        classes = os.listdir(self.path)
        for cls in classes:
            if not os.path.isdir(os.path.join(self.path, cls)):
                continue
            pickle_name = cls + ".pickle"
            self.pickles.append(pickle_name)
            if os.path.exists(os.path.join(self.path, pickle_name)) and not force:
                print("{}: already present. Skipping".format(pickle_name))
            else:
                dataset = self.import_class(cls)
                try:
                    with open(os.path.join(self.path, pickle_name), "wb") as pkl:
                        pickle.dump(dataset, pkl, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print("{}: Error while saving".format(f))
        return self.pickles

    def merge(self, train_size, valid_size=0):
        pickle_files = glob.glob(self.path +"/*.pickle")
        num_classes  = len(pickle_files)

        train_size_per_class = train_size // num_classes
        valid_size_per_class = valid_size // num_classes
        # Creating training datasets
        train_x_dataset = np.ndarray(
          shape=(train_size, self.size, self.size),
          dtype=np.float32
        )
        train_y_dataset = np.ndarray(train_size, dtype=np.int32)
        # Creating validation dataset
        valid_x_dataset = np.ndarray(
          shape=(valid_size, self.size, self.size),
          dtype=np.float32
        )
        valid_y_dataset = np.ndarray(valid_size, dtype=np.int32)

        train_counter_start, train_counter_end = 0, train_size_per_class
        valid_counter_start, valid_counter_end = 0, valid_size_per_class

        for label, pkl_f in enumerate(pickle_files):
            with open(pkl_f, 'rb') as pf:
                letters = pickle.load(pf)
                np.random.shuffle(letters)

                if valid_size_per_class > 0:
                    valid_x_dataset[valid_counter_start:valid_counter_end, :, :] = letters[:valid_size_per_class, :, :]
                    valid_y_dataset[valid_counter_start:valid_counter_end] = label
                    valid_counter_start += valid_size_per_class
                    valid_counter_end += valid_size_per_class

            train_x_dataset[train_counter_start:train_counter_end, :, :] = letters[valid_size_per_class:valid_size_per_class + train_size_per_class, :, :]
            train_y_dataset[train_counter_start:train_counter_end] = label
            train_counter_start += train_size_per_class
            train_counter_end += train_size_per_class

        return train_x_dataset, train_y_dataset, valid_x_dataset, valid_y_dataset

    def export_pickle_merged(self, out, train_size, valid_size=0, force=False):
        if os.path.exists(out) and not force:
            print("{}: already exists. Skipping".format(out))
        else:
            t_x_db, t_y_db, v_x_db, v_y_db = self.merge(train_size, valid_size)
            t_x_db, t_y_db = self.reformat(t_x_db, t_y_db)
            if v_x_db is not None:
                v_x_db, v_y_db = self.reformat(v_x_db, v_y_db)
            with open(out, "wb") as pf:
                pickle.dump({
                  'train': (t_x_db, t_y_db),
                  'valid': (v_x_db, v_y_db)
                }, pf, pickle.HIGHEST_PROTOCOL)

    def reformat(self, x, y, shape_x=None, y_label=None):
        if not shape_x:
            shape_x = (-1, self.size, self.size, 1)
        if not y_label:
            y_label = self.classes
        x = x.reshape(shape_x).astype(np.float32)
        y = (np.arange(y_label) == y[:, None]).astype(np.float32)
        return x, y

    def load(self, f):
        with open(f, "rb") as fp:
            return pickle.load(fp)

class Database(dict):
    def __init__(self, db1, db2):
        ddb1 = self.load(db1)
        ddb2 = self.load(db2)

        self["train"] = {"x": ddb1["train"][0], "y": ddb1["train"][1]}
        self["valid"] = {"x": ddb1["valid"][0], "y": ddb1["valid"][1]}
        self["test"]  = {"x": ddb2["train"][0], "y": ddb2["train"][1]}

    def load(self, f):
        with open(f, "rb") as fp:
            return pickle.load(fp)

    def train(self, t):
        return self["train"][t]

    def test(self, t):
        return self["test"][t]

    def valid(self, t):
        return self["valid"][t]
