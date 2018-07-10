from keras.models import Model, load_model
from keras.layers import Dense, Dropout
from keras.metrics import top_k_categorical_accuracy
from keras.utils import to_categorical

from misc.AttentionWeightedAverage import *
from misc.funcs import *

from textgenrnn import textgenrnn
from textgenrnn.utils import textgenrnn_encode_sequence

import numpy as np
from numpy.random import randint, permutation, seed

from matplotlib import pyplot as plt

import os

import json

#
#
#

class Highlighter:

    def __init__(self, maxlen=40, sample_stride=3):
        # DEBUG: changing maxlen and sample_stride will break model
        self.authors = []
        self.authors_dict = {}
        self.num_authors = 0
        self.paths = {} # will store paths to text files, keys will be the authors
        self.texts = [] # DEBUG: necessary?
        self.texts_dict = {}
        self.vocab = textgenrnn().vocab
        # length of text snippets analyzed
        self.maxlen = maxlen # recommend to leave at default to utilize transfer learning

        self.num_samples_per_author = None
        self.sample_stride = sample_stride
        self.encoded_texts = [] # the texts we are working with, encoded in the format needed for the model
        self.labels = [] # labels corresponding to the training data




    def build_model(self, textgenrnn_weights_path=0, num_authors=None):
        # DEBUG: make this changeable by user
        """define and compile the core highlighter model"""

        if self.num_authors is None and num_authors is None:
            print('Error: must specify number of authors.')
            return

        # define model
        if textgenrnn_weights_path is 0:
            self._tg_rnn = textgenrnn(textgenrnn_weights_path)
            self.vocab = self._tg_rnn.vocab

        else:
            self._tg_rnn = textgenrnn()
        self._tg_model = self._tg_rnn.model
        self._input = self._tg_model.layers[0].input
        self._tg_out = self._tg_model.layers[-2].output
        self._tg_drop = Dropout(rate=0.5)(self._tg_out)
        self._classification = Dense(units=self.num_authors,
                                     activation='softmax',
                                     name='classification')(self._tg_drop)
        self.model = Model(inputs=self._input, outputs=self._classification)

        # compile model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['acc', top_3_acc])




    def conv_text(self, text):
        # DEBUG: requires build_model to be run first
        """convert given text into format required by model"""
        return textgenrnn_encode_sequence(text, vocab=self.vocab, maxlen=self.maxlen)[0]




    def load_author_texts(self, path, num_samples_per_author=None, verbose=True, rand_seed=0):
        # DEBUG: requires build_model to be run first
        """
        Trains on a corpus of several authors, located in path (name of folder with data).
        Files must be .txt and folders must have tree structure given by
            ├--path
            |  ├--author_1
            |     ├--writing_1_by_author_1.txt
            |     ├--writing_2_by_author_1.txt
            |     ├-- ...
            |  ├--author_2
            |     ├--writing_1_by_author_2.txt
            |     ├--writing_2_by_author_2.txt
            |     ├-- ...
            |  ├-- ...
        Author names will be read in as they appear in the
            corresponding folder's name.
        Note that "path" must end in a '/' to indicate it is a folder.
        """

        if verbose:
            print('loading text and author data...')
        # index the authors and the paths of their works
        for author in os.listdir(path):
            self.authors += [author]
            self.paths[author] = []
            for filename in os.listdir(path + author):
                if '.txt' in filename:
                    self.paths[author].append(path + author + '/' + filename)
        self.authors.sort()
        self.num_authors = len(self.authors)

        # stores the indices associated to each author
        self.authors_dict = { author:i for i,author in enumerate(self.authors)}
        # store the joined texts associated to each author
        for author in self.authors:
            all_text_by_author = []
            for path in self.paths[author]:
                all_text_by_author += [open(path,'r',encoding='latin').read()]
            all_text_by_author = '\n'.join(all_text_by_author)
            self.texts_dict[author] = all_text_by_author


        # list version of texts_dict
        self.texts = [self.texts_dict[author] for author in self.authors] # DEBUG: necessary?

        if verbose:
            print('done.')

        if verbose:
            print('found authors:')
            for author,text in zip(self.authors,self.texts):
                print('\t'+author)
                print('\t\tlength chars in text: {0:8d}'.format(len(text)))
            print()

        # now that we have the authors loaded we can build the model
        try:
            self.model
        except:
            self.build_model(self.num_authors)

        # we want to have the same amount of samples for each author
        if verbose:
            print('processing text...')
        if num_samples_per_author is None and self.num_samples_per_author is None:
            author_lengths = [int((len(text) - self.maxlen) / self.sample_stride) + 1 for text in self.texts]
            self.num_samples_per_author = min(author_lengths)
        else:
            self.num_samples_per_author = num_samples_per_author
        self.encoded_texts = []
        self.labels = []
        for author in self.authors:
            text = self.texts_dict[author]
            i = self.authors_dict[author]
            size = len(text)
            # if the size of the text is the minimum of all texts, sample uniformly
            if size == self.num_samples_per_author:
                for j in range(self.num_samples_per_author):
                    idx = j * self.sample_stride
                    self.encoded_texts.append(text[idx:idx+self.maxlen])
                    self.labels.append(i)
            # otherwise sample randomly the same number of times
            else:
                for _ in range(self.num_samples_per_author):
                    idx = randint(0, size-self.maxlen)
                    self.encoded_texts.append(text[idx:idx+self.maxlen])
                    self.labels.append(i)
        # convert text snippets into format required by model
        self.encoded_texts = list(map(lambda t : textgenrnn_encode_sequence(t, self.vocab, self.maxlen)[0],
                                      self.encoded_texts))
        self.encoded_texts = np.array(self.encoded_texts)
        self.labels = np.array(self.labels)
        # seed the random number generator
        seed(rand_seed)
        # randomly shuffle the data
        self.shuffle_idx = list(permutation(self.num_samples_per_author * self.num_authors))
        self.encoded_texts = self.encoded_texts[self.shuffle_idx]
        self.labels = self.labels[self.shuffle_idx]
        # convert labels to one-hot
        self.labels = to_categorical(self.labels, self.num_authors)

        if verbose:
            print('done.')




    def train(self, epochs, validation_split=0.2, verbose=True):
        """train the classifier model on loaded data"""

        # make sure we have data loaded before training
        if self.labels is None or self.encoded_texts is None:
            print('No texts loaded -- load them with the load_training_data method.')
            return
        else:
            # train the model
            print('training...')
            self.model.fit(x=self.encoded_texts, y=self.labels,
                           epochs=epochs,
                           validation_split=validation_split,
                           verbose=verbose)
            print('done.')




    def highlight(self, text, padding=False):
        """
        Scans given text to find the most likely author at each point in text.
        padding=True ensures that the output is the same length as the original text.
        """
        try:
            self.model
            snippets = []
            for i in range(len(text)-self.maxlen+1):
                converted_snippet = self.conv_text(text[i:i+self.maxlen])
                snippets.append(converted_snippet)
            snippets = np.array(snippets)
            if padding==True:
                snippets = np.concatenate([[snippets[0]]*int(np.floor(self.maxlen/2)), snippets])
                snippets = np.concatenate([snippets, [snippets[-1]]*int(np.ceil(self.maxlen/2))])
            return self.model.predict(snippets)
        except:
            print('Error: make sure model is loaded and that text length is longer than maxlen.')
            return -1




    def classify(self, text):
        """Tries to predict the author of the given text."""
        highlighting = self.highlight(text)
        avgs = np.mean(highlighting, axis=0)
        author = self.authors[np.argmax(avgs)]
        return author



    def plot_highlights(self, text, authors=None):
        """
        Renders a plot of the highlighting intensity for each author as the
        text progresses character by character.
        """
        highlighting = self.highlight(text, padding=True)
        if authors is None:
            authors = self.authors
        for i in range(len(self.authors)):
            plt.plot(highlighting[:,i], label=self.authors[i]);
            plt.legend()
        plt.show()




    def save_model(self, filepath_model, filepath_vars=0):
        """
        Saves the model's variables in a JSON and the model architecture/weights in a .hdf5 file.
        """
        try:
            # write model weights and architecture to file
            self.model.save(filepath_model)
        except:
            print('Error saving model file. Check saving path.')

        if type(filepath_vars) is str:
            vars = {}
            vars['authors'] = self.authors
            vars['authors_dict'] = self.authors_dict
            vars['num_authors'] = self.num_authors
            vars['paths'] = self.paths
            # vars['texts'] = self.texts
            # vars['texts_dict'] = self.texts_dict
            vars['maxlen'] = self.maxlen
            vars['num_samples_per_author'] = self.num_samples_per_author
            vars['sample_stride'] = self.sample_stride
            vars['encoded_texts'] = self.encoded_texts.tolist()
            vars['labels'] = self.labels.tolist()
            vars['vocab'] = self.vocab

            try:
                # write variables to file
                with open(filepath_vars, 'w') as outfile:
                    json.dump(vars, outfile)
            except:
                print('Error saving var file. Check saving path.')




    def load_model(self,
                   filepath_model, filepath_vars=0):
        """
        Load the model's variables and weights from a JSON file.
        """
        try:
            # load the model from file
            self.model = load_model(filepath_model,
                                    custom_objects={"AttentionWeightedAverage":AttentionWeightedAverage,
                                                    "top_3_acc":top_3_acc})
            # compile the model
            self.model.compile(loss='categorical_crossentropy',
                               optimizer='adamax',
                               metrics=['acc', top_3_acc])
        except:
            print('Error loading model file from specified path.')
            return -1
        if type(filepath_vars) is str:
            try:
                # load the variables from file
                with open(filepath_vars, 'r', encoding='latin') as json_file:
                    vars = json.load(json_file)
                self.authors = vars['authors']
                self.authors_dict = vars['authors_dict']
                self.num_authors = vars['num_authors']
                self.paths = vars['paths']
                self.texts = vars['texts']
                self.texts_dict = vars['texts_dict']
                self.maxlen = vars['maxlen']
                self.num_samples_per_author = vars['num_samples_per_author']
                self.sample_stride = vars['sample_stride']
                self.encoded_texts = np.array(vars['encoded_texts'])
                self.labels = np.array(vars['labels'])
                self.vocab = vars['vocab']
            except FileNotFoundError:
                print('Couldn\'t find model variables. Loading default vocab.')
                self.vocab = textgenrnn().vocab
                if self.num_authors == 0:
                    self.num_authors = K.int_shape(self.model.output)[-1]
                if len(self.authors) == 0:
                    self.authors = [str(i) for i in range(self.num_authors)]
