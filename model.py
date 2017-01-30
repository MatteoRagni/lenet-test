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


import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, config):
        r"""
        Inizializza un nuovo modello di LeNet-5. Lo fa usando 4 funzioni.
         * legge la configurazione, impostando le dimensioni delle variabili
         * crea le variabili
         * costruisce mediante le variabili il modello da ottimizzare
         * costruisce la funzione ottimizzata

        Tutto il modello è costruito internamente ad un grafo (che è quello di default)
        """
        # Graph (space for declaration in C++ lib)
        self.graph = tf.Graph()

        self.read_config(config)
        self.define_variables()
        self.define_model()
        self.define_optimizer(config["learning.rate"])
        self.define_prediction()


    def define_variables(self):
        r"""
        Crea le variabili (che differiscono dalle dimensioni espresse in read_config), che
        rappresentano uno spazio di memoria nella libreria che si occupa della ottimizzazione.
        """
        with self.graph.as_default():
            # Layer 1: Convolutional Variables
            self.l1_w = self.weight(self.weight_1, name="conv.1.w")
            self.l1_b = self.bias(self.bias_1, name="conv.1.b")

            # Layer 3: Convolutional Variables
            self.l3_w = self.weight(self.weight_2, name="conv.2.w")
            self.l3_b = self.bias(self.bias_2, name="conv.2.b")

            # Layer 5: Hidden fully connected layer Variables
            self.l5_w = self.weight(self.weight_3, name="fc.1.w")
            self.l5_b = self.bias(self.bias_3, name="fc.1.b")

            # Layer 7: Readout layer
            self.l7_w = self.weight(self.weight_4, name="fc.2.w")
            self.l7_b = self.bias(self.bias_4, name="fc.2.b")

        self.weight_to_summary("l1.w", self.l1_w)
        self.weight_to_summary("l3.w", self.l3_w)


    def define_model(self):
        r"""
        Definisce il modello vero e proprio: un grafo computazionale a partire da x fino a label

        INPUT = [no_batches, 28, 28, 1]
         1. convolutional layer
          * w = [5, 5, 1, 32]
          * b = [32]
         2. max pooling layer
          * stride = 2
          * input image = [28, 28, 32] -> output image = [14, 14, 32]
         3. convolutional layer
          * w = [5, 5, 32, 64]
          * b = [64]
         4. max pooling layer
          * stride = 2
          * input image = [7, 7, 64] -> output image = [7, 7, 64]
          * prima di passare al layer successivo: reshape([7, 7, 64] -> [7 * 7 * 64])
         5. fully connected layer
          * w = [7 * 7 * 64, 1024]
          * b = [1024]
         6. dropout layer
         7. fully connected layer
          * w = [1024, 10]
          * b = [10]
        OUTPUT = [no_batches, class]
        """
        with self.graph.as_default():
            # Layer 1: Convolutional layer
            with tf.name_scope("layer.1"):
                self.l1 = tf.nn.relu(self.convolution(self.x, self.l1_w, name="layer.1.convolution") +
                  self.l1_b, name="layer.1.relu")

            # Layer 2: Max pooling layer
            with tf.name_scope("layer.2"):
                self.l2 = self.max_pool(self.l1, name="layer.2.maxpooling")

            # Layer 3: Convolutional layer
            with tf.name_scope("layer.3"):
                self.l3 = tf.nn.relu(self.convolution(self.l2, self.l3_w, name="layer.3.convolution") +
                  self.l3_b, name="layer.3.relu")

            # Layer 4: Max pooling layer
            with tf.name_scope("layer.4"):
                self.l4 = self.max_pool(self.l3, name="layer.4.maxpooling")
                self.l4_flat = tf.reshape(self.l4, self.l4reshape, name="layer.4.reshaping")

            # Layer 5: Fully connected layer
            with tf.name_scope("layer.5"):
                self.l5 = self.fully_connect(self.l4_flat, self.l5_w, self.l5_b,
                  name="layer.5.fully.connected")

            # Layer 6: Dropout layer
            with tf.name_scope("layer.6"):
                self.l6 = tf.nn.dropout(self.l5, self.l6_dropout)

            # Layer 7: Fully connected output layer
            with tf.name_scope("layer.7"):
                self.l7 = self.fully_connect(self.l6, self.l7_w, self.l7_b,
                 name="layer.7.fully.connected")
        return self.l7

    def define_optimizer(self, learning_rate=1e-4):
        r"""
        Definisce la operazione di ottimizzazione. Questa operazione è a tutti gli effetti parte del
        grafo della rete neurale
        """
        with self.graph.as_default():
            with tf.name_scope("output.layer"):
                self.cross_entropy = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits(logits=self.l7, labels=self.label),
                  name="cross.entropy")

                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

                # Aggiungo la cross.entropy al sommario per tensorboard
                tf.summary.scalar("cross.entropy", self.cross_entropy)
        return self.cross_entropy, self.optimizer


    def define_prediction(self):
        r"""
        Anche la predizione fa parte del grafo della rete neurale. In questo caso si calcola la
        media rispetto alla serie di label in ingresso.
        """
        with self.graph.as_default():
            self.predict_label = tf.argmax(self.l7, 1)
            self.predict_prob  = tf.nn.softmax(self.l7)
            self.correct_label = tf.argmax(self.label, 1)
            self.accuracy = tf.reduce_mean(
              tf.cast(
                tf.equal(
                  self.predict_label,
                  self.correct_label
                ),
                tf.float32
              )
            )

            # Aggiungo accuracy all'elenco del sommario per tensorboard
            tf.summary.scalar("accuracy", self.accuracy)
        return self.accuracy, self.predict_label

    #
    # Funzioni di supporto, nulla di troppo interessante. Fare riferimento ai commenti delle singole
    # funzioni
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
        # Layer 4: fully connected reshaping
        self.l4reshape = [-1, config["fc.1.in"]]
        # Configuration for layer 5
        self.weight_3 = [
          config["fc.1.in"],
          config["fc.1.out"]
        ]
        self.bias_3 = [config["fc.1.out"]]
        # Layer 6: Dropout layer
        # Configuration for layer 7
        self.weight_4 = [
          config["fc.1.out"],
          config["fc.2.out"]
        ]
        self.bias_4 = [config["fc.2.out"]]

        self.gpu = config["use.gpu"]

        # Configuration for input layer
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32,
              shape=[None, config["image.size"], config["image.size"], config["depth.1"]],
              name="input")
            self.label = tf.placeholder(tf.float32,
              shape=[None, config["fc.2.out"]],
              name="labels")
            self.l6_dropout = tf.placeholder(tf.float32, name="dropout.keep-probability")


    def weight(self, shape, name=""):
        r"""
        Ritorna una nuova variabile peso iniziailizzata con valore a media nulla preso da
        una distribuzione normale con deviazione standard = 0.1. Richiede una shape in ingresso.
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


    def bias(self, shape, name=""):
        r"""
        Ritorna una variabile che rappresenta un bias inizializzato a 0.1, con una shape
        specificiata. Essendo parte di un relu, si inizializza con un valore leggermente positivo
        """
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)


    def convolution(self, x, w, name=""):
        r"""
        Ritorna la convoluzione tra una funzione composta x e una variabile w
        """
        if self.gpu:
            return tf.nn.conv2d(x, w, strides=self.conv_stride, padding=self.conv_padding, name=name)

        return tf.nn.conv2d(x, w, strides=self.conv_stride, padding=self.conv_padding,
          use_cudnn_on_gpu=False, name=name)


    def max_pool(self, x, name=""):
        r"""
        Ritorna il max pooling di una funzione composta x
        """
        return tf.nn.max_pool(x, ksize=self.mp_size, strides=self.mp_stride,
          padding=self.mp_padding, name=name)


    def fully_connect(self, x, w, b, name=""):
        r"""
        Ritorna un fully connected layer (con ReLU - rectified linear unit) di una funzione composta
        x e una variabile w e un peso b come logit.
        """
        return tf.nn.relu(tf.matmul(x, w) + b, name=name)

    def weight_to_summary(self, name, var, batch=1):
        r"""
        Aggiunge i pesi come sommario. Li va a spezzare sulla base di input e di output.
        Quindi in generale per un peso di dimensione (X, X, 5, 4) produce 20 immagini
        con nome: name.in.2.out.3 ad esempio.
        """
        with self.graph.as_default():
            with tf.name_scope("weights.visualization"):
                shape = var.get_shape()
                for i in range(0, shape[2]):
                    for o in range(0, shape[3]):
                        l_name = ("%s.in.%d.out.%d" % (name, i, o))
                        tf.summary.image(l_name, tf.reshape(var[:, :, i:i+1, o:o+1], [1,5,5,1]))
                        # TODO Hardcoded [5, 5]: bisogna sistemarlo.


class bcolors:
    r"""
    source: http://stackoverflow.com/a/287944/2319299
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_training_information(batch_size, step, accuracy, correct_label, predict_label, predict_prob):
    r"""
    Una funzione che stampa a schermo alcune infrmazioni riguardanti lo stato attuale
    del training. Per esmpio mostra per ogni elemento del corrente batch size:
     * step
     * accuratezza media
     * per ogni elemento della batch:
       * indice
       * label corretta
       * label predetta
       * probabilità percentuale predizione
       * (se predizione sbagliata) percentuale predizione della label corretta
    usando il colore rosso per le label sbagliate e il colore verde per le label corrette
    """
    abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    print(bcolors.BOLD + ("Step: %8d - Accuracy: %3.1f%%" % (step, accuracy * 100.0)) + bcolors.ENDC)
    for i in range(0, batch_size):
        color = bcolors.OKGREEN if correct_label[i] == predict_label[i] else bcolors.FAIL
        other_prob = "" if correct_label[i] == predict_label[i] else (" (%3.5f%%)" % predict_prob[i][correct_label[i]])
        print(color +
          ("  %3d: (%s) -> (%s) %3.5f%%" % (i, abc[correct_label[i]],
            abc[predict_label[i]], predict_prob[i][predict_label[i]] * 100.0)) +
            other_prob +
          bcolors.ENDC)
