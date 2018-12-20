# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#Baseado em https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from unicodedata import normalize

from sklearn.manifold import TSNE
from os import listdir
from os.path import isfile, join

import collections
import math

import numpy as np
import tensorflow as tf

import pickle

import nltk
import random
import matplotlib.pyplot as plt

random_seed = 10

random.seed(random_seed)
np.random.seed(random_seed)
FOLDER = 'text_files/entrada/concessÆo_da_ordem'

def normalize_text(text):
    text = normalize('NFKD', text).encode('ASCII', 'ignore')

    final_text = text.lower()

    if type(final_text) == bytes:
        final_text = final_text.decode('UTF-8')

    return final_text

def doc_generator_to_token_generator(doc_generator):
    for doc in doc_generator:
        tokens = nltk.word_tokenize(doc)
        for token in tokens:
            yield token


def get_existent_text_files(folder, name_only=False):

    files_only = [f for f in listdir(folder) if isfile(join(folder, f))]
    txts_only = [f for f in files_only if '.txt' in f]
    if name_only:
        names_only = [f.replace('.txt', '') for f in txts_only]
        return names_only
    else:
        return [join(folder, c) for c in txts_only]

def get_text_generator(folder):
    print('FOLDER', folder)
    files_in_folder = get_existent_text_files(folder)
    for file in files_in_folder:
        yield open(file, 'r').read()

def build_dictionaries(n_words, folder):
    batches_gen = get_text_generator(folder)
    token_generator = doc_generator_to_token_generator(batches_gen)

    print('building dictionaries')

    forward_dictionary = {}
    for token in token_generator:
        if token in forward_dictionary:
            forward_dictionary[token] += 1
        else:
            forward_dictionary[token] = 1

    keys = forward_dictionary.keys()
    values = list(forward_dictionary.values())
    forward_dictionary_list = [(k, values[i]) for i, k in enumerate(keys)]
    sorted_forward_dictionary = sorted(forward_dictionary_list, key=lambda x: x[1], reverse=True)
    final_forward_dictionary = {'UNK': -1}
    index = 0
    for token, amount in sorted_forward_dictionary[:n_words]:
        final_forward_dictionary[token] = index
        index += 1

    reverse_dictionary = dict(zip(final_forward_dictionary.values(), final_forward_dictionary.keys()))

    return [final_forward_dictionary, reverse_dictionary]


def data_generator(token_generator, forward_dictionary):
    for token in token_generator:
        if token in forward_dictionary:
            yield forward_dictionary[token]
        else:
            yield 0


def create_token_index_generator(folder, forward_dictionary):
    batch_gen = get_text_generator(folder)
    token_gen = doc_generator_to_token_generator(batch_gen)
    data_gen = data_generator(token_gen, forward_dictionary)
    return data_gen


def make_batch(data_gen, batch_size, num_skips, skip_window):
    global data_index
    assert(batch_size%num_skips == 0)
    assert(num_skips <= 2*skip_window)

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2*skip_window + 1
    buffer = collections.deque(maxlen=span)
    buff_elem_list = []
    for i in range(span):
        try:
            new_elem = data_gen.__next__()
            buff_elem_list.append(new_elem)
        except:
            break

    if len(buff_elem_list) == 0:
        return [], []

    buffer.extend(buff_elem_list)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        try:
            new_buff_elem = data_gen.__next__()
            buffer.append(new_buff_elem)
        except:
            return [], []

    return batch, labels

def plot_with_labels(low_dims_embs, labels, filename='tsne.png'):
    assert low_dims_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18,18))
    axes = plt.gca()
    axes.set_xlim([-60, 60])
    axes.set_ylim([-60, 60])

    print('Creating figure')
    for i, label in enumerate(labels):
        x, y = low_dims_embs[i, :]
        plt.annotate(label, xy=(x, y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')

    plt.savefig(filename)
    print('Figure saved')


def fetch_word_vector(word, dictionary, final_embeddings):
    try:
        index = dictionary[word]
    except KeyError:
        index = dictionary['UNK']

    vector = final_embeddings[index, :]
    return vector


def euclid_dist(v1, v2):
    diff = v1 - v2
    return np.sqrt(np.dot(diff, np.transpose(diff)))


def get_closest_word(vector, words_to_avoid, dictionary, final_embeddings):
    words = dictionary.keys()
    best_word = 0
    best_distance = 10000000000000000000

    for w in words:
        w_v = fetch_word_vector(w, dictionary, final_embeddings)
        new_dist = euclid_dist(vector, w_v)
        if new_dist < best_distance and w not in words_to_avoid:
            best_distance = new_dist
            best_word = w

    return best_word


def create_and_train_word2vec_model(vocabulary_size, forward_dictionary, reverse_dictionary, show_figure=False):

    data_gen = create_token_index_generator(FOLDER, forward_dictionary)

    batch_size = 128
    embedding_size = 128
    skip_window = 1
    num_skips = 2

    print('Starting')
    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64

    graph = tf.Graph()

    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

    num_steps = 100001

    with tf.Session(graph=graph) as session:
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = make_batch(data_gen, batch_size, num_skips, skip_window)
            if len(batch_inputs) == 0:
                data_gen = create_token_index_generator(forward_dictionary)
                batch_inputs, batch_labels = make_batch(data_gen, batch_size, num_skips, skip_window)

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    print('Average loss at step ', step, ':', average_loss)
                    average_loss = 0

        final_embeddings = normalized_embeddings.eval()

    try:
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)

        results_dictionary = {'final_embeddings': final_embeddings,
                              'reverse_dictionary': reverse_dictionary,
                              'low_dim_embs': low_dim_embs,
                              'dictionary': forward_dictionary}
        return results_dictionary
    except ImportError:
        print('Please install sklearn, matplotlib and scipy to show emdeddings.')
        return None


def create_word_embeddings():
    vocabulary_size = 10000
    [forward_dictionary, reverse_dictionary] = build_dictionaries(vocabulary_size, FOLDER)
    results_dictionary = create_and_train_word2vec_model(vocabulary_size, forward_dictionary, reverse_dictionary)
    return results_dictionary


def create_and_save_word_embeddings():
    results_dictionary = create_word_embeddings()
    out = open('trained_objects/word2vec_out.pickle', 'wb')
    pickle.dump(results_dictionary, out)
    out.close()