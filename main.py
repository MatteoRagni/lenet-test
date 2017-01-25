import sys
sys.path.insert(0, '.')

import dataset


## Dataset Donwload

baseUrl    = 'http://commondatastorage.googleapis.com/books1000/'
downloader = dataset.Downloader(baseUrl, "dataset")

downloader.download('notMNIST_large.tar.gz').extract('notMNIST_large.tar.gz')
downloader.download('notMNIST_small.tar.gz').extract('notMNIST_small.tar.gz')
