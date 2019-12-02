import numpy as np
import tensorflow as tf

class RNN_Edge(tf.keras.Model):
    def __init__(self, english_vocab_size):
        super(RNN_Edge, self).__init__()
        self.english_vocab_size = english_vocab_size

        self.rnn_size = 32
        self.learning_rate = 0.01
        self.batch_size = 100
        self.embedding_size = 50

        self.enEmbedding = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size, input_length=self.english_window_size)
