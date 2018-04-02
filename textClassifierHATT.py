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
import argparse
import math
import pickle

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
from keras.models import Model, load_model
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
        
        # Save posts_by_blog
        with open('/usr0/home/mamille2/posts_by_blog.pkl', 'wb') as f:
            pickle.dump(posts_by_blog, f)

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
        indices = np.arange(len(data))
        X_train, X_test, y_train, y_test, inds_train, inds_test = train_test_split(data, labels, indices, test_size=test_size, random_state=0)

        X_train, X_dev, y_train, y_dev, inds_train, inds_dev = train_test_split(X_train, y_train, inds_train, test_size=test_size, random_state=0)

        # Save dev indices
        with open('/usr0/home/mamille2/dev_inds.pkl', 'wb') as f:
            pickle.dump(inds_dev, f)
    
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

        model.summary()
        self.model = model


    def train_model(self, X_train, y_train, X_dev, y_dev, epochs=10, batch_size=16):
        self.model.fit(X_train, y_train, validation_data=(X_dev, y_dev),
              epochs=epochs, batch_size=batch_size)


    def load_model(self, model_name, dirpath):

        #if model_name == "default":
        #    # Load first model in dir
        #    model_name = os.listdir(dirpath)[0]
        
        model_path = os.path.join(dirpath, f"blog_predict_identity_cat_{model_name}.h5")
        self.model = load_model(model_path, custom_objects={'AttLayer': AttLayer})


    def save_model(self, dirpath):
        outpath = os.path.join(dirpath, f"blog_predict_identity_cat_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')}.h5")
        print(f"Saving model to {outpath}...", end=' ')
        self.model.save(outpath)
        print('done.')

    
    def predict(self, X):
        preds = self.model.predict(X)
        return preds
    

    def get_attention_weights(self, X):
        get_layer_output = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[3].output])
        att_w = self.model.layers[4].get_weights()[0]

        weight_list = []

        batch_size = 16
        start = 0
        end = start + batch_size

        for i in range(math.floor(len(X)/batch_size)):
            batch = X[start:end]
            out = get_layer_output([batch, 0])[0]
            
            # Maybe could just get output of attention layer
            for j in range(batch_size):
                eij = np.tanh(np.dot(out[j], att_w))
                ai = np.exp(eij)
                weights = ai/np.sum(ai)
                weight_list.append(weights)

            start = end
            end = start + batch_size

        # Save attention weights
        with open('/usr0/home/mamille2/tumblr_attn_test.pkl', 'wb') as f:
            pickle.dump(weight_list, f)
        pdb.set_trace()

        return weight_list


def main():

    parser = argparse.ArgumentParser(description="Train and run hierarchical attention network")
    parser.add_argument('--load-model', nargs='?', dest='model_name')
    args = parser.parse_args()

    model_dirpath = '/usr0/home/mamille2/tumblr/models/'

    # Load, preprocess data
    dh = DataHandler(max_num_words=100000)
    dh.load_data(
            '/usr0/home/mamille2/tumblr/data/list_descriptions_100posts.pkl', 
            '/usr0/home/mamille2/tumblr/data/textposts_recent100_100posts.pkl')
    X_train, X_dev, X_test, y_train, y_dev, y_test = dh.process_data()

    dh.print_info(X_train, X_dev, X_test, y_train, y_dev, y_test)

    han = HAN()
    han.load_model(args.model_name, model_dirpath)

    if args.model_name:
        
        # Load model
        print("Loading model...", end=' ')
        sys.stdout.flush()
        han.load_model(args.model_name, model_dirpath)
        print("done.")
        sys.stdout.flush()

        # Get, save attention weights
        print("Getting attention weights...", end=" ")
        sys.stdout.flush()
        attn_weights = han.get_attention_weights(X_dev)
        

        print('done.')
        sys.stdout.flush()

    else:

        # Build model
        print("Building model...", end=' ')
        sys.stdout.flush()
        han.build_model(dh.max_num_words, dh.embedding_dim, dh.max_post_length, dh.max_num_posts)
        print('done.')
        sys.stdout.flush()

        # Train model
        print("Training model...", end=' ')
        sys.stdout.flush()
        han.train_model(X_train, y_train, X_dev, y_dev)
        print('done.')
        sys.stdout.flush()

        # Save model
        han.save_model(model_dirpath)

if __name__ == '__main__':
    main()
