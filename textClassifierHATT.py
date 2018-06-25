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
import html
from pprint import pprint

os.environ['KERAS_BACKEND']='theano'

# Use CPU
#os.environ['THEANO_FLAGS'] = 'device=cpu'

# Use GPU
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['THEANO_FLAGS'] = 'device=cuda'
os.environ['THEANO_FLAGS'] = 'floatX=float32'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
#os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score


class SaveBestModel(ModelCheckpoint):
    """ Keras callback with custom save function """

    def __init__(self, model_container, 
                 monitor='val_loss', verbose=0,
                 save_best_only=False,
                 mode='auto', period=1):
        self.model_container = model_container
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model'
                                  % (epoch + 1, self.monitor, self.best,
                                     current))
                        self.best = current
                        self.model_container.save_model()
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model' % (epoch + 1))

                self.model_container.save_model()


class DataHandler():
    """ For loading and preprocessing data.  """

    def __init__(self, data_dirpath,
                name=datetime.datetime.now().strftime('%Y-%m-%dT%H-%M'),
                max_num_posts=100, 
                max_post_length=200, 
                max_num_words = 50000,
                test_dev_split = 0.1):

        self.max_num_posts = max_num_posts
        self.max_post_length = max_post_length
        self.max_num_words = max_num_words
        self.test_dev_split = test_dev_split
        self.descs = None
        self.posts = None
        self.tids = None
        self.tids_split = {}
        self.cats = None
        self.X = {}
        self.y = {}
        self.name = name
        self.data_dirpath = data_dirpath # where will save processed data

    def load_data(self, descs_filepath, posts_filepath):

        # Load descriptions
        self.descs = pd.read_pickle(descs_filepath)
        self.posts = pd.read_pickle(posts_filepath)
        self.tids = sorted(self.descs['tumblog_id'].tolist())

    def process_data(self, input_colname='body_toks_str_no_titles', outcome_colname='all', save=True):
        """ Preprocesses data and returns vectorized form """
        print("Preprocessing data...", end=" ")
        sys.stdout.flush()

        # Get text posts
        posts_by_blog = [[p for p in self.posts[self.posts['tumblog_id']==tid][input_colname].tolist()] for tid in self.tids] # list of 100 posts/user
        all_posts = [p for posts in posts_by_blog for p in posts]

        # Tokenize text posts
        tokenizer = Tokenizer(num_words=self.max_num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”')
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
        if outcome_colname == 'all':
            cols = self.descs.columns.tolist()
            self.cats = [c for c in cols[cols.index('parsed_blog_description')+1:] if not c.endswith('terms')]

        else:
            self.cats = [outcome_colname] # can only handle 1 colname
        
        labels = np.array(list(zip(*[self.descs[cat] for cat in self.cats])))
#        if len(self.cats) == 1:
#            labels = to_categorical(np.array(labels, dtype=int)) 

        # Shuffle, split into train/dev/test
        test_size = int(self.test_dev_split * len(data))
        X_train, self.X['test'], y_train, self.y['test'], \
                tids_train, self.tids_split['text'] = \
            train_test_split(data, labels, self.tids, test_size=test_size, random_state=0)

        self.X['train'], self.X['dev'], self.y['train'], self.y['dev'], tids_train, self.tids_split['dev'] = train_test_split(X_train, y_train, tids_train, test_size=test_size, random_state=0)

        # Save vectorized data
        if save:
            vectorized_datapath = os.path.join(self.data_dirpath, f"{self.name}_preprocessed_data.pkl")

            dict_save = self.__dict__.copy()
            dict_save.pop('descs')
            dict_save.pop('posts')

            with open(vectorized_datapath, 'wb') as f:
                pickle.dump(dict_save, f)

        print("done.")
        sys.stdout.flush()

        print(f"Saved preprocessed data to {vectorized_datapath}")        
        sys.stdout.flush()
        

    def load_processed_data(self, name):
        """ Load preprocessed DataHandler object """

        vectorized_datapath = os.path.join(self.data_dirpath, f"{name}_preprocessed_data.pkl")
        with open(vectorized_datapath, 'rb') as f:
            tmp = pickle.load(f)

        self.__dict__.update(tmp)

    def print_info(self):
    
        # Number of samples
        print()
        print("# training instances: {}".format(len(self.X['train'])))
        print("# dev instances: {}".format(len(self.X['dev'])))
        print("# test instances: {}".format(len(self.X['test'])))
        print()

        # Shapes of tensors
        print("Shape of training feature tensor: {}".format(self.X['train'].shape))
        print("Shape of training labels tensor: {}".format(self.y['train'].shape))
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

    def __init__(self, base_dirpath, name=None):
        self.model = None
        if name:
            self.model_name = name
        else:
            self.model_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M')
        self.base_dirpath = base_dirpath
        self.model_dirpath = os.path.join(base_dirpath, 'models', self.model_name)
        if not os.path.exists(self.model_dirpath):
            os.mkdir(self.model_dirpath)
        self.output_dirpath = os.path.join(base_dirpath, 'output', self.model_name)
        if not os.path.exists(self.output_dirpath):
            os.mkdir(self.output_dirpath)
        self.embeddings_paths = {
            'tumblr_halfday': (os.path.join(base_dirpath, 'data/recent100_100posts_embeds.npy'), 300),
            'tumblr_recent100_fasttext': (os.path.join(base_dirpath, 'data/blog_descriptions_100posts_embeds.npy'), 100),
            'tumblr_recent100_300dim': (os.path.join(base_dirpath, 'data/blog_descriptions_100posts_300dim.npy'), 300),
            'tumblr_recent100_tags': (os.path.join(base_dirpath, 'data/textposts_100posts_tags.npy'), 100),
            } # name: (fpath, ndims)


    def _build_embedding_layer(self, vocab_size, max_post_length, embeddings):

        # Load embeddings
        vocab_embed = np.load(self.embeddings_paths[embeddings][0])

        embedding_dim = self.embeddings_paths[embeddings][1]
        embedding_layer = Embedding(
                            vocab_size,
                            embedding_dim,
                            weights = [vocab_embed],
                            input_length = max_post_length,
                            trainable=True
                        )

        return embedding_layer


    def build_model(self, vocab_size, max_post_length, max_num_posts, 
        embeddings='tumblr_recent100_300dim', n_outcomes=14):
        
        embedding_layer = self._build_embedding_layer(vocab_size, max_post_length, embeddings)

        post_input = Input(shape=(max_post_length,), dtype='int32')
        embedded_sequences = embedding_layer(post_input)
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        l_dense = TimeDistributed(Dense(200))(l_lstm)
        l_att = AttLayer()(l_dense)
        post_encoder = Model(post_input, l_att)

        blog_input = Input(shape=(max_num_posts,max_post_length), dtype='int32')
        blog_encoder = TimeDistributed(post_encoder)(blog_input)
        l_lstm_post = Bidirectional(GRU(100, return_sequences=True))(blog_encoder)
        l_dense_post = TimeDistributed(Dense(200))(l_lstm_post)
        l_att_post = AttLayer()(l_dense_post)

        #preds = Dense(2, activation='softmax')(l_att_post)
        preds = Dense(n_outcomes, activation='sigmoid')(l_att_post)
        model = Model(blog_input, preds)

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        self.post_encoder = post_encoder
        self.model = model


    def train_model(self, X_train, y_train, X_dev, y_dev, epochs=10, batch_size=16):
        callbacks = [SaveBestModel(self, monitor='val_acc', save_best_only=True, verbose=1)]
        self.model.fit(X_train, y_train, validation_data=(X_dev, y_dev),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks)


    def load_model(self, model_name):

        model_path = os.path.join(self.model_dirpath, "classifier.h5")
        post_encoder_path = os.path.join(self.model_dirpath, "post_encoder.h5")
        self.model_name = model_name
        self.model = load_model(model_path, custom_objects={'AttLayer': AttLayer}) # Keras function
        self.post_encoder = load_model(post_encoder_path, custom_objects={'AttLayer': AttLayer})


    def save_model(self):

        # Save main model
        outpath = os.path.join(self.model_dirpath, "classifier.h5")
        self.model.save(outpath)

        # Save post encoder
        outpath = os.path.join(self.model_dirpath, "post_encoder.h5")
        self.post_encoder.save(outpath)

    
    def predict(self, X):
        preds = self.model.predict(X)
        return preds
    

    def post_attention_weights(self, X):
        get_layer_output = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[3].output])
        #get_layer_output = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[4].output])
        att_w = self.model.layers[4].get_weights()[0]

        weight_list = []

        batch_size = 16
        start = 0
        end = start + batch_size

        for i in range(math.floor(len(X)/batch_size)):
            batch = X[start:end]
            out = get_layer_output([batch, 0])[0]
            #weight_list.extend(out)
            
            for j in range(batch_size):
                eij = np.tanh(np.dot(out[j], att_w))
                ai = np.exp(eij)
                weights = ai/np.sum(ai)
                weight_list.append(weights)

            start = end
            end = start + batch_size

        # Save attention weights
        outpath = os.path.join(self.output_dirpath, "attention_weights.pkl")
        with open(outpath, 'wb') as f:
            pickle.dump(weight_list, f)

        return weight_list


    def word_attention_weights(self, X):
        get_layer_output = K.function([self.post_encoder.layers[0].input, K.learning_phase()], [self.post_encoder.layers[3].output])
        att_w = self.post_encoder.layers[4].get_weights()[0]

        weight_list = []

        batch_size = 16
        start = 0
        end = start + batch_size

        for i in range(math.floor(len(X)/batch_size)):
            batch = X[start:end]
            
            for j in range(batch_size):
                out = get_layer_output([batch[j], 0])[0]
                eij = np.tanh(np.dot(out, att_w))
                ai = np.exp(eij)
                weights = ai/np.sum(ai)
                weight_list.append(weights)

            start = end
            end = start + batch_size

        # Save attention weights
        outpath = os.path.join(self.output_dirpath, f"word_attention_weights.pkl")
        with open(outpath, 'wb') as f:
            pickle.dump(weight_list, f)

        return weight_list


    def _set_scores(self, preds, actual):
        """ Returns set measures of precision, recall and f1.
            Might be able to do the same thing with scikit learn average='macro' with eg precision_score.
            """
        
        precisions = []
        recalls = []

        total_prec = precision_score(actual, preds, average='weighted')
        total_rec = recall_score(actual, preds, average='weighted')
        total_f1 = f1_score(actual, preds, average='weighted')
        
        # Old way--flawed
        #for pred, act in zip(preds, actual):
        #    pred_labels = set(np.flatnonzero(pred)) # binary encoding to indices
        #    actual_labels = set(np.flatnonzero(act))
        #    correct = pred_labels.intersection(actual_labels)
        #    
        #    # precision
        #    if len(pred_labels) == 0:
        #        if len(actual_labels) == 0:
        #            prec = 1.0 # Case where no labels predicted and actually are no labels
        #        else: 
        #            prec = 0.0
        #    else:
        #        prec = len(correct)/len(pred_labels)
        #    precisions.append(prec)
        #    
        #    # recall
        #    if len(actual_labels) == 0:
        #        if len(pred_labels) == 0:
        #            rec = 1.0 # Case where no labels predicted and actually are no labels
        #        else:
        #            rec = 0.0
        #    else:
        #        rec = len(correct)/len(actual_labels)
        #    recalls.append(rec)
        #    
        #total_prec = np.mean(precisions)
        #total_rec = np.mean(recalls)
        #total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec)
        
        return {'precision': total_prec,
                'recall': total_rec,
                'f1': total_f1}


    def _category_scores(self, preds, y, cats):
        metrics = {'kappa': cohen_kappa_score,
                    'precision': precision_score,
                    'recall': recall_score,
                    'f1': f1_score}
        scores = {}

        for cat, i in zip(cats, list(range(preds.shape[1]))):
            pred_col = preds.T[i]
            actual_col = y.T[i]
            scores[cat] = {}
            
            for name, scorer in metrics.items():
                scores[cat][name] = scorer(pred_col, actual_col)
        
        return scores


    def evaluate(self, X, X_name, X_tids, y, cats):
        """ 
            Returns prec, recall, f1 and kappa for each category and overall.
            Args:
                * cats: list of names of categories, in the order of labeled instances in y
        """

        preds = self.predict(X)

        preds[preds>=0.5] = True
        preds[preds<0.5] = False

        # Save predictions
        pred_path = os.path.join(self.output_dirpath, f"{X_name}_preds.pkl")
        pred_df = pd.DataFrame()
        pred_df['tumblog_id'] = X_tids
        for cat, pred, actual in zip(cats, preds.T, y.T):
            pred_df[f'pred_{cat}'] = pred
            pred_df[f'actual_{cat}'] = actual
            pred_df[f'actual_{cat}'] = pred_df[f'actual_{cat}'].astype(float)

        pred_df.to_pickle(pred_path)

        scores = {}

        # Per-category scores
        scores['category'] = self._category_scores(preds, y, cats)

        # Aggregate scores
        scores['set'] = self._set_scores(preds, y)

        # Save scores
        metrics = ['precision', 'recall', 'f1', 'kappa']
        outlines = [['all'] + [scores['set'][m] for m in metrics[:-1]], \
                    *[[c] + [scores['category'][c][m] for m in metrics] for c in cats]]
        outlines = pd.DataFrame(outlines, columns=['category'] + metrics)
        if not os.path.exists(self.output_dirpath):
            os.mkdir(self.output_dirpath)
        outpath = os.path.join(self.output_dirpath, "scores.csv")
        outlines.to_csv(outpath, index=False)

        return pred_df, scores


    def _color_attention(self, wts, total_max, total_min):
        """ Returns 0-1 for highlighting """
        
        scale = total_max-total_min
        vals = [(wt-total_min)/scale for wt in wts]
        return vals

    
    def _color_post(self, wds, wts, total_max, total_min):
        """ Returns html with colored background for word attention """
    
        vals = self._color_attention(wts, total_max, total_min)

    
        #return ''.join([f"<span style='background-color: rgba(255,0,0,{val})'>{html.escape(wd)}</span>&nbsp" for val, wd in zip(vals, wds)])

        html_str = ""
        for i, (val, wd) in enumerate(list(zip(vals, wds))):
            wd_str = f"<span style='background-color: rgba(255,0,0,{val})'>{html.escape(wd)}</span>&nbsp"
            if i % 15 == 0:
                wd_str += '<br>'
            html_str += wd_str

        return html_str
    

    def attention_visualization(self, post_weights, word_weights, tids, dh, descs_path, posts_path, pred_df,
        multiply_weights=False, by_category=False):
        """ Max word weights are set from max word weight per blog """
        
        # Load data
        dh.descs = pd.read_pickle(descs_path)
        dh.posts = pd.read_pickle(posts_path)

        # Select posts by blog
        sel_posts = {}
        sel_weighted_posts = {}
        sel_wd_weights = {}
        sel_post_wd_weights = {}
        top_post_wd_weights = {}
        for tid in tids:
            sel_posts[tid] = dh.posts[dh.posts['tumblog_id']==tid]['body_toks_str_no_titles'].tolist()

        # Assign weights to sentences
        for i in range(len(post_weights)):
            tid = tids[i]
            sel_weighted_posts[tid] = [(post_weights[i][j], post) for j, post in enumerate(sel_posts[tid])] # assuming posts are in order of weights (might want to save out order in DataHandler object)
            
            # Word weights
            post_wd_wts = word_weights[i]
            posts = sel_posts[tid]
        
            sel_wd_weights[tid] = []

            for j in range(len(posts)):
                wds = posts[j].split()
                if multiply_weights:
                    wts = post_weights[i][j] * post_wd_wts[j][:len(wds)]
                else:
                    wts = post_wd_wts[j][:len(wds)]
                sel_wd_weights[tid].append(list(zip(wts, wds)))

            post_wd_wts, _ = list(zip(*sel_weighted_posts[tid]))
            sel_post_wd_weights[tid] = list(zip(post_wd_wts, sel_wd_weights[tid])) # tid: [(post_wt, [(wd_wt, wd), ...]), ...]

            # Take top 5 and bottom 5 posts
            sel_post_wd_weights[tid] = sorted(sel_post_wd_weights[tid], reverse=True)
            sel_post_wd_weights[tid] = sel_post_wd_weights[tid][:5]+ sel_post_wd_weights[tid][-5:]
            top_post_wd_weights[tid] = sel_post_wd_weights[tid][0]

        # Format posts as HTML string
        post_strings = []
        top_post_strings = []
        for tid in sel_post_wd_weights.keys():
            top_post_strings.append(top_post_wd_weights[tid])
            if len([el[0] for post in sel_wd_weights[tid] for el in post]) == 0:
                continue
            max_wt = max([el[0] for post in sel_wd_weights[tid] for el in post])
            min_wt = min([el[0] for post in sel_wd_weights[tid] for el in post])
    
            wd_wts = sel_wd_weights[tid]

            posts_str = ''
            for i, (post_wt, wd_wt_list) in enumerate(sel_post_wd_weights[tid]):
                if len(wd_wt_list) == 0:
                    continue

                wts, wds = list(zip(*wd_wt_list))
                post_str = self._color_post(wds, wts, max_wt, min_wt)
                posts_str += post_str + '<br><br>'

                if i == 4:
                    posts_str += '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<br><br>'

            post_strings.append(posts_str)

        post_str_df = pd.DataFrame(list(zip(tids, post_strings, top_post_strings)), columns=['tumblog_id', 'ranked_post_str', 'top_post'])
        #post_str_df['ranked_post_str'] = post_str_df['ranked_post_str'].str.wrap(100)# word wrap
        #post_str_df = post_str_df.sample(50, random_state=7) # downsample

        # Get description, predicted and gold category labels
        columns = ['tumblog_id', 'restr_segments_25', 'ranked_post_str'] + [f'{cat}_terms' for cat in dh.cats]
        merged = pd.merge(dh.descs, dh.posts, on='tumblog_id')
        merged = pd.merge(merged, post_str_df, on='tumblog_id').loc[:, columns]

        # Merge with predictions, actual category values
        merged = pd.merge(merged, pred_df, on='tumblog_id')
        slice_preds = merged.loc[:, [f'pred_{cat}' for cat in dh.cats]]
        preds_true = slice_preds.apply(lambda x: x > 0)
        merged['predicted_categories'] = preds_true.apply(lambda x: list(slice_preds.columns[x.values]), axis=1)

        slice_act = merged.loc[:, [f'actual_{cat}' for cat in dh.cats]]
        actual_true = slice_act.apply(lambda x: x > 0)
        merged['actual_categories'] = actual_true.apply(lambda x: list(slice_act.columns[x.values]), axis=1)

        # Drop duplicates
        merged.drop_duplicates(subset=['tumblog_id'], inplace=True)

        # Save top posts
        outpath = os.path.join(self.output_dirpath, "top_posts.pkl")
        merged.to_pickle(outpath)

        # Separate by category
        if by_category:
            cats = list(dh.cats) + ['all', 'none']
        else:
            cats = ['all']

        for cat in cats:

            # Select data based on category
            if cat == 'none':
                mask = [len(tup[0])==0 or len(tup[1])==0 for tup in zip(merged['predicted_categories'], merged['actual_categories'])]
                cat_df = merged[mask]

            elif cat != 'all':
                mask = [(f'pred_{cat}' in tup[0]) or (f'actual_{cat}' in tup[1]) for tup in zip(merged['predicted_categories'], merged['actual_categories'])]
                cat_df = merged[mask]

            # Select columns
            sel_cols = ['tumblog_id', 'restr_segments_25', 'predicted_categories', 'actual_categories', 'ranked_post_str'] + [f'{cat}_terms' for cat in dh.cats] + [f'pred_{cat}' for cat in dh.cats] + [f'actual_{cat}' for cat in dh.cats]
            cat_df = cat_df.loc[:, sel_cols]

            if len(cat_df) > 50:
                cat_df = cat_df.sample(50, random_state=7) # downsample

            if not cat in ['all', 'none']:
                cat_df.sort_values([f'pred_{cat}', f'actual_{cat}'], inplace=True, ascending=False) # error when try to sort by multiple list columns
                #cat_df.sort_values([f'{cat}_terms', 'predicted_categories'], inplace=True, ascending=False) # error when try to sort by multiple list columns

            # Save table as HTML
            pd.set_option('display.max_colwidth', 500)
            s = cat_df.style.set_properties(**{'vertical-align': 'top', 'border': '1px solid gray', 'border-collapse': 'collapse',
                'word-wrap': 'break-word'})
            html_tab = s.render()
            outpath = os.path.join(self.output_dirpath, f"{cat.replace('/', '-')}_attn_viz.html")
            with open(outpath, 'w') as f:
                f.write(html_tab)
        

    def load_attention_weights(self):
        path = os.path.join(self.output_dirpath, "attention_weights.pkl")
        with open(path, 'rb') as f:
            weight_list = pickle.load(f)

        word_path = os.path.join(self.output_dirpath, "word_attention_weights.pkl")
        with open(word_path, 'rb') as f:
            word_weight_list = pickle.load(f)

        return word_weight_list, weight_list


def main():

    parser = argparse.ArgumentParser(description="Train and run hierarchical attention network")
    parser.add_argument('--base_dirpath', nargs='?', help="Path to parent directory with data, where should save models and output directories", default='/usr0/home/mamille2/tumblr/')
    parser.add_argument('--dataset-name', nargs='?', dest='dataname', help="Name to save preprocessed data to")
    parser.add_argument('--model-name', nargs='?', dest='model_name', help="Name to save model to")
    parser.add_argument('--input', nargs='?', dest='input_colname', help="Name of column with input features. Should be a string with space-separated features.")
    parser.add_argument('--outcome', nargs='?', dest='outcome_colname', help="Name of column/s to predict")
    parser.add_argument('--embeddings', nargs='?', dest='embeddings', help="Name of pretrained embeddings to load", default='tumblr_recent100_300dim')
    parser.add_argument('--epochs', nargs='?', dest='n_epochs', help="Number of epochs to train", default=100, type=int)
    parser.add_argument('--load-model', nargs='?', dest='load_model')
    parser.add_argument('--load-data', nargs='?', dest='load_dataname', help="Name of preprocessed data to load")
    parser.add_argument('--load-attention', dest='load_attn', action='store_true')
    parser.set_defaults(outcome_colname=None)
    args = parser.parse_args()

    base_dirpath = args.base_dirpath
    data_dirpath = os.path.join(base_dirpath, 'data')
    #descs_path = os.path.join(base_dirpath, 'data/list_descriptions_100posts.pkl')
    #posts_path = os.path.join(base_dirpath, 'data/textposts_recent100_100posts.pkl')
    descs_path = os.path.join(base_dirpath, 'data/blog_descriptions_recent100_100posts.pkl')
    #posts_path = os.path.join(base_dirpath, 'data/textposts_100posts.pkl')
    posts_path = os.path.join(base_dirpath, 'data/textposts_100posts.pkl')

    if args.outcome_colname:
        outcome_colname = args.outcome_colname
    else:
        outcome_colname = 'all'

    # Load, preprocess data
    if args.load_dataname:
        # Load preprocessed, vectorized data
        dh = DataHandler(data_dirpath)
        
        print("Loading preprocessed data...", end=' ')
        sys.stdout.flush()
        dh.load_processed_data(args.load_dataname)
        print("done.")
        sys.stdout.flush()
    
    else:
        print("Loading data...", end=' ')
        sys.stdout.flush()
        if args.dataname:
            dh = DataHandler(data_dirpath, name=args.dataname, max_num_words=100000)
        else:
            dh = DataHandler(data_dirpath, max_num_words=100000)
        dh.load_data(descs_path, posts_path)
        print("done.")
        sys.stdout.flush()
        dh.process_data(input_colname=args.input_colname, outcome_colname=outcome_colname)
        
        dh.print_info()


    if args.load_model:
        
        han = HAN(base_dirpath, name=args.load_model)

        # Load model
        print("Loading model...", end=' ')
        sys.stdout.flush()
        han.load_model(args.load_model)
        print("done.")
        sys.stdout.flush()

    else:

        han = HAN(base_dirpath, name=args.model_name)

        # Build model
        print("Building model...", end=' ')
        sys.stdout.flush()
        if args.outcome_colname:
            han.build_model(dh.max_num_words, dh.max_post_length, dh.max_num_posts, embeddings=args.embeddings, n_outcomes=1)
        else:
            han.build_model(dh.max_num_words, dh.max_post_length, dh.max_num_posts, n_outcomes=len(dh.cats))
        print('done.')
        han.model.summary()
        sys.stdout.flush()

        # Train model
        print("\nTraining model...", end=' ')
        sys.stdout.flush()
        han.train_model(dh.X['train'], dh.y['train'], dh.X['dev'], dh.y['dev'], epochs=args.n_epochs)
        print('done.')
        sys.stdout.flush()

        # Save model
        han.save_model()

    # Evaluate
    print("Evaluating model...", end=" ")
    sys.stdout.flush()
    pred_df, scores = han.evaluate(dh.X['dev'], 'dev', dh.tids_split['dev'], dh.y['dev'], dh.cats)
    print("done.")
    sys.stdout.flush()
    #pprint(scores)

    if args.load_attn:
        print("Loading attention weights...", end=' ')
        sys.stdout.flush()
        word_attn_weights, post_attn_weights = han.load_attention_weights()
        print("done.")
        sys.stdout.flush()

    else:
        # Save attention weight visualization
        print("Getting attention weights...", end=" ")
        sys.stdout.flush()
        post_attn_weights = han.post_attention_weights(dh.X['dev'])
        word_attn_weights = han.word_attention_weights(dh.X['dev'])
        print('done.')
        sys.stdout.flush()

    print("Making attention weight visualization...", end=" ")
    sys.stdout.flush()
    han.attention_visualization(post_attn_weights, word_attn_weights, dh.tids_split['dev'], dh, descs_path, posts_path, pred_df, multiply_weights=True, by_category=True)
    print('done.')
    sys.stdout.flush()

if __name__ == '__main__':
    main()
