import numpy as np
import tensorflow as tf
import csv
import regex as re

def prepreprocess(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['story'], row['target'])

def extract_names(query):
    midpoint = query.find(",")
    names = [query[2:midpoint-1], query[midpoint+3:-2]]
    return names

def paint(sentence, query):
    names = extract_names(query)
    s1 = sentence.replace(names[0], "person0")
    s2 = s1.replace(names[1], "person1")
    s3 = s2.replace(".","").lower()
    return s3[:-1].split(" ")

def read_data(file_name):
    data = []
    labels = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            z = paint(row['story'], row['query'])
            if len(z) == 6:
                data.append(z)
                labels.append(row['target'])
    return data, labels


def build_vocab(sentences):
    """
    DO NOT CHANGE

    Builds vocab from list of sentences

    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set(tokens)))

    vocab =  {word:i for i,word in enumerate(all_words)}

    return vocab

def convert_to_id(vocab, sentences):
    """
    DO NOT CHANGE

    Convert sentences to indexed

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.array(list(map(lambda x: np.array(list(map(lambda y: vocab[y], x))), sentences)))

def get_data(file):
    """
    returns data, labels
    """
    data, labels = read_data(file)

    labeltypes = list(set(labels))
    labelDict = {k: v for v, k in enumerate(labeltypes)}
    newlabels = list(map(lambda x: labelDict[x], labels))
    labels = tf.one_hot(newlabels, depth=len(labeltypes), dtype=tf.int8)

    vocab = build_vocab(data)
    idse = convert_to_id(vocab, data)

    # print(newlabels)
    # print(labels)

    return idse, labels, vocab, labelDict

def snag_data(file, labelDict, vocab):
    data, labels = read_data(file)

    newlabels = list(map(lambda x: labelDict[x], labels))
    # print(newlabels)
    labels = tf.one_hot(newlabels, depth=len(labelDict), dtype=tf.int8)

    idse = convert_to_id(vocab, data)

    return idse, np.array(labels)
