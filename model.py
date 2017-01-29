#!/usr/bin/env python3
#-*-coding:utf-8-*-

import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, config):
        r"""
        Inizializza un nuovo modello di LeNet-5. Lo fa usando 4 funzioni.
         * legge la configurazione, impostando le dimensioni delle variabili
         * crea le variabili
         * costruisce mediante le variabili il modello da ottimizzare

        Tutto il modello è costruito internamente ad un grafo (che è quello di default)
        """

        # Graph (space for declaration in C++ lib)
        self.graph = tf.Graph()

        self.read_config(config)
        self.define_variables()
        self.define_model()



    def define_variables(self):
        with self.graph.as_default():
            # Layer 1: Convolutional Variables
            self.l1_w = self.weight(self.weight_1, name="conv.1.w")
            self.l1_b = self.bias(self.bias_1, name="conv.1.b")

            # Layer 3: Convolutional Variables
            self.l3_w = self.weight(self.weight_2, name="conv.2.w")
            self.l3_b = self.bias(self.bias_2, name="conv.2.b")

            # Layer 5: Hidden fully connected layer Variables
            self.l5_w = self.weight(self.weight_3, name="fc.1.w")
            self.l5_b = self.weight(self.bias_3, name="fc.1.b")

            # Layer 7: Readout layer
            self.l7_w = self.weight(self.weight_4, name="fc.2.w")
            self.l7_b = self.weight(self.bias_4, name="fc.2.b")


    def define_model(self):
        with self.graph.as_default():
            # Layer 1: Convolutional layer
            self.l1 = tf.nn.relu(self.convolution(self.x, self.l1_w) + self.l1_b, name="layer.1.convolution")
            # Layer 2: Max pooling layer
            self.l2 = self.max_pool(self.l1, name="layer.2.maxpooling")
            # Layer 3: Convolutional layer
            self.l3 = tf.nn.relu(self.convolution(self.l2, self.l3_w) + self.l3_b, name="layer.3.convolution")
            # Layer 4: Max pooling layer
            self.l4 = self.max_pool(self.l3, name="layer.4.maxpooling")
            # Layer 5: Fully connected layer
            self.l5 = self.fully_connected(self.l4, self.l5_w, self.l5_b, name="layer.5.fully.connected")
            # Layer 6: Dropout layer
            self.l6 = tf.nn.dropout(self.l5, self.l6_dropout)
            # Layer 7: Fully connected output layer
            self.y = self.fully_connected(self.l6, self.l7_w, self.l7_b, name="layer.7.fully.connected")
        return self.y

    #
    # Support function
    #
    def read_config(self, config):
        r"""
        Questa funzione legge un dizionario di configurazione con tutte le dimensioni necessarie
        a poter ricostruire la architettura di LeNet-5.

        Questa struttura è stata scelta per semplificare la spiegazione, non necessariamente il modo
        migliore di definire una NN
        """
        # Convolution common configuration
        self.conv_stride  = config["conv.stride"]
        self.conv_padding = config["conv.padding"]
        # Max Pooling common configuration
        self.mp_size    = config["pool.size"]
        self.mp_stride  = config["pool.stride"]
        self.mp_padding = config["pool.padding"]

        # Configuration for input layer
        self.x = tf.placeholder(tf.float32,
          shape=[None, config["image.size"], config["image.size"], config["depth.1"]])
        self.label = tf.placeholder(tf.float32,
          shape=[None, config["fc.2.out"]])

        # Configuration for layer 1
        self.weight_1 = [
          config["patch.1"],
          config["patch.1"],
          config["depth.1"],
          config["depth.2"]
        ]
        self.bias_1 = [config["depth.2"]]
        # Configuration for layer 3
        self.weight_2 = [
          config["patch.2"],
          config["patch.2"],
          config["depth.2"],
          config["depth.3"]
        ]
        self.bias_2 = [config["depth.3"]]
        # Configuration for layer 5
        self.weight_3 = [
          config["fc.1.in"],
          config["fc.1.out"]
        ]
        self.bias_3 = [config["fc.1.out"]]
        # Layer 6: Dropout layer
        self.l6_dropout = tf.placeholder(tf.float32, name="dropout.keep-probability")
        # Configuration for layer 7
        self.weight_4 = [
          config["fc.1.out"],
          config["fc.2.out"]
        ]
        self.bias_3 = [config["fc.2.out"]]

    def weight(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


    def bias(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))


    def convolution(x, w):
        return tf.nn.conv2d(x, w, strides=self.conv_stride, padding=self.conv_padding)


    def max_pool(x, name=""):
        return tf.nn.max_pool(x, ksize=self.mp_size, strides=self.mp_stride, padding=self.mp_padding, name=name)

    def fully_connect(x, w, b, name=""):
        return tf.nn.relu(tf.matmul(x, w) + b, name=name)
