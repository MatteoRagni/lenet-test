#!/usr/bin/env python3

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
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

class Downloader:
    def __init__(self, baseUrl, distDir):
        self.baseUrl = baseUrl
        self.baseDir = distDir
        if not os.path.isdir(distDir):
            os.mkdir(distDir)

    def dwn_progress(self, count, blkSize, totSize):
        percent = int(count * blkSize * 100 / totSize)
        if percent % 5 == 0:
            print(("\r" * 4) + ("%3d%%" % percent), end="")

    def download(self, filename, force=False):
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
        root = os.path.splitext(os.path.splitext(filename)[0])[0]
        filename = os.path.join(self.baseDir, filename)
        if os.path.isdir(root) and not force:
            print("{}: already extracted. Skipping.".format(filename))
        else:
            tar = tarfile.open(filename)
            tar.extractall(path=self.baseDir)
            tar.close()
        return self

class Dataset:
    def __init__(self):
        pass
