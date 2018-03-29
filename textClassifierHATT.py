# author - Richard Liao 
# Dec 26 2016
# modified by Michael Miller Yoder
# 2018

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import sys
import os
import pdb
import datetime

os.environ['KERAS_BACKEND']='theano'
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['THEANO_FLAGS'] = 'device=cuda'
#os.environ['THEANO_FLAGS'] = 'device=gpu'
os.environ['THEANO_FLAGS'] = 'floatX=float32'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from sklearn.model_selection import train_test_split


class DataHandler():
    """ For loading and preprocessing data """

    def __init__(self, max_num_posts=100, 
                max_post_length=200, 
                max_num_words = 50000,
                embedding_dim = 300,
                test_dev_split = 0.1):

        self.max_num_posts = max_num_posts
        self.max_post_length = max_post_length
        self.max_num_words = max_num_words
        self.embedding_dim = embedding_dim
        self.test_dev_split = test_dev_split
        self.descs = None
        self.posts = None
        self.tids = None

    def load_data(self, descs_filepath, posts_filepath):

        print("Loading data...", end=' ')
        sys.stdout.flush()

        # Load descriptions
        self.descs = pd.read_pickle(descs_filepath)
        self.posts = pd.read_pickle(posts_filepath)
        self.tids = sorted(self.descs['tumblog_id'].tolist())

        print('done.')
        sys.stdout.flush()

    def process_data(self, vectorize=True):
        """ Preprocesses data and returns vectorized form if set """
        print("Preprocessing data...", end=" ")
        sys.stdout.flush()

        # Get text posts
        posts_by_blog = [[p for p in self.posts[self.posts['tumblog_id']==tid]['body_str_no_titles'].tolist()] for tid in self.tids] # list of 100 posts/user

        all_posts = [p for posts in posts_by_blog for p in posts]

        # Tokenize text posts
        tokenizer = Tokenizer(num_words=self.max_num_words,   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”')
        tokenizer.fit_on_texts(all_posts)
        self.word_index = tokenizer.word_index
        self.vocab = list(self.word_index.keys())[:self.max_num_words]

        data = np.zeros((len(posts_by_blog), self.max_num_posts, self.max_post_length), dtype='int32')

        for i, posts in enumerate(posts_by_blog):
            for j, post in enumerate(posts):
                if j < self.max_num_posts:
                    wordTokens = text_to_word_sequence(post)
                    k=0
                    for _, word in enumerate(wordTokens):
                        if k < self.max_post_length and word in self.word_index and self.word_index[word] < self.max_num_words:
                            data[i,j,k] = tokenizer.word_index[word]
                            k=k+1                    

        # Prepare description categories (labels)
        cats = self.descs.columns.tolist()[-8:]
        labels = np.array(list(zip(*[self.descs[cat] for cat in cats])))
        #labels = to_categorical(np.array(labels, dtype=int))

        # Shuffle, split into train/dev/test
        test_size = int(self.test_dev_split * len(data))
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=test_size)
    
        print("done.")
        sys.stdout.flush()
        
        return X_train, X_dev, X_test, y_train, y_dev, y_test

    def print_info(self, X_train, X_dev, X_test, y_train, y_dev, y_test):
    
        # Number of samples
        print()
        print("# training instances: {}".format(len(X_train)))
        print("# dev instances: {}".format(len(X_dev)))
        print("# test instances: {}".format(len(X_test)))
        print()

        # Shapes of tensors
        print("Shape of training feature tensor: {}".format(X_train.shape))
        print("Shape of training labels tensor: {}".format(y_train.shape))
        print()


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],))
        #self.trainable_weights = [self.W]
        self.W = self.add_weight(name='kernel', 
                                  shape=(input_shape[-1],),
                                  initializer='normal',
                                  trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class HAN():
    """ Hierarchical Attention Network """

    def __init__(self):
        self.model = None
        self.embeddings_paths = {
            'tumblr_halfday': '/usr0/home/mamille2/tumblr/data/recent100_100posts_embeds.npy'
            }

    def _build_embedding_layer(self, vocab_size, embedding_dim,max_post_length, embeddings):

        # Load embeddings
        #print("Loading pretrained embedding weights...", end=' ')
        #sys.stdout.flush()
        vocab_embed = np.load(self.embeddings_paths[embeddings])
        embedding_layer = Embedding(
                            vocab_size,
                            embedding_dim,
                            weights = [vocab_embed],
                            input_length = max_post_length,
                            trainable=True
                        )

        return embedding_layer

    def build_model(self, vocab_size, embedding_dim, max_post_length, max_num_posts, embeddings='tumblr_halfday'):
        
        embedding_layer = self._build_embedding_layer(vocab_size, embedding_dim, max_post_length, embeddings)

        post_input = Input(shape=(max_post_length,), dtype='int32')
        embedded_sequences = embedding_layer(post_input)
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        l_dense = TimeDistributed(Dense(200))(l_lstm)
        l_att = AttLayer()(l_dense)
        postEncoder = Model(post_input, l_att)

        blog_input = Input(shape=(max_num_posts,max_post_length), dtype='int32')
        blog_encoder = TimeDistributed(postEncoder)(blog_input)
        l_lstm_post = Bidirectional(GRU(100, return_sequences=True))(blog_encoder)
        l_dense_post = TimeDistributed(Dense(200))(l_lstm_post)
        l_att_post = AttLayer()(l_dense_post)
        preds = Dense(8, activation='sigmoid')(l_att_post)
        model = Model(blog_input, preds)

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        return model


def save_model(model, dirpath):
    outpath = os.path.join(dirpath, f"blog_predict_identity_cat_{datetime.datetime.now().strftime('%Y-%m-%DT%H-%M')}.h5")
    print(f"Saving model to {outpath}...", end=' ')
    model.save(outpath)
    print('done.')


def main():

    model_dirpath = '/usr0/home/mamille2/tumblr/models/'

    # Load, preprocess data
    dh = DataHandler(max_num_words=100000)
    dh.load_data(
            '/usr0/home/mamille2/tumblr/data/list_descriptions_100posts.pkl', 
            '/usr0/home/mamille2/tumblr/data/textposts_recent100_100posts.pkl')
    X_train, X_dev, X_test, y_train, y_dev, y_test = dh.process_data()

    dh.print_info(X_train, X_dev, X_test, y_train, y_dev, y_test)

    # Build model
    print("Building model...", end=' ')
    sys.stdout.flush()
    han = HAN()
    model = han.build_model(dh.max_num_words, dh.embedding_dim, dh.max_post_length, dh.max_num_posts)
    print('done.')
    sys.stdout.flush()

    model.summary()

    # Train model
    print("Training model...", end=' ')
    sys.stdout.flush()
    model.fit(X_train, y_train, validation_data=(X_dev, y_dev),
          epochs=10, batch_size=16)
    print('done.')
    sys.stdout.flush()

    # Save model
    save_model(model, model_dirpath)

if __name__ == '__main__':
    main()
