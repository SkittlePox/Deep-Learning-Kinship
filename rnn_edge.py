import numpy as np
import tensorflow as tf
from preprocess import *
import math

class RNN_Edge(tf.keras.Model):
    def __init__(self, english_vocab_size, labelCount):
        super(RNN_Edge, self).__init__()
        self.english_vocab_size = english_vocab_size

        self.rnn_size = 32
        self.learning_rate = 0.01
        self.batch_size = 20
        self.embedding_size = 50

        self.enEmbedding = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size)
        self.encoder = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(labelCount, activation='softmax')

    @tf.function
    def call(self, encoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        enc = self.enEmbedding(encoder_input)
        output, state1, state2 = self.encoder(enc)

        probs = self.dense(state1)

        return probs

    def loss_function(self, prbs, labels):
        # sum = 0.0
        # print(labels)
        # print(np.shape(prbs))
        # for i in range(np.shape(prbs)[0]):
        #     # print(prbs[0][i])
        #     maxval = np.argmax(prbs[i])
        #     if (maxval != labels[i]):
        #         # print(probabilities[i][maxval])
        #         # print(math.log(probabilities[i][maxval]))
        #         sum += math.log(prbs[i][maxval])
        #         # print(maxval, labels[i])
        #
        # return sum / np.shape(prbs)[0] * -1
        # print(np.shape(labels))
        # print(np.shape(prbs))
        return(tf.keras.losses.categorical_crossentropy(labels, prbs))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def test(model, test_inputs, test_labels):
    logits = model.call(test_inputs)
    return(model.accuracy(logits, test_labels))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch.
    '''

    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    for i in range(0, len(train_labels), model.batch_size):
        if len(train_labels) - i < 0:
            break
        with tf.GradientTape() as tape:
            logits = model.call(train_inputs[i:i+model.batch_size])
            loss = model.loss_function(logits, train_labels[i:i+model.batch_size])

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return None

def main():
    training_data, training_labels, vocab, labelDict = get_data('data/1.1_train.csv')
    testing_data, testing_labels = snag_data('data/1.1_test.csv', labelDict, vocab)

    model = RNN_Edge(len(vocab), len(labelDict))


    print("Pre-train Testing:")
    print(test(model, testing_data, testing_labels))
    print("Training")
    train(model, training_data, training_labels)
    print("Post-train Testing")
    print(test(model, testing_data, testing_labels))


if __name__ == "__main__":
    main()
