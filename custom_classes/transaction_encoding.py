from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import string

class CreateWordTokenizer(object):
    def __init__(self, df_transactions, included_columns, max_dict_size=None, external_dict_path = None):
        self.df_transactions = df_transactions
        self.included_columns = included_columns
        self.max_dict_size = max_dict_size
        self.external_dict_path = external_dict_path
        self._select_text_for_training()
        self._train_tokenizer()
        
    def _select_text_for_training(self):
        self.all_text = []
        for column in self.included_columns:
            self.all_text.extend(list(self.df_transactions[column]))
        
    def _train_tokenizer(self):
        self.tokenizer = Tokenizer(num_words=None,
                      #filters = '/',
                      filters='!"#$%()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      oov_token=None)
        
        if not(self.external_dict_path == None):
            self._load_external_dict()
            combined_text = list(self.embeddings) + self.all_text
            self.tokenizer.fit_on_texts(combined_text)            
        else:
            self.tokenizer.fit_on_texts(self.all_text)
        print len(self.tokenizer.word_counts)
        
    def _load_external_dict(self):
        # load the whole embedding into memory
        self.embeddings = dict()
        
        f = open(self.external_dict_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings[word] = coefs
        f.close()
        print('Loaded %s word vectors from external dictionary' % len(self.embeddings))


class TransactionConverter(object):
    def __init__(self, word_tokenizer):
        self._init_BA_ID_tokenizer()
        self.word_tokenizer = word_tokenizer.tokenizer        
        
    def _init_BA_ID_tokenizer(self):
        # used to convert an arbitrary bank account ID (IBAN) to a sequence of integers
        all_characters = string.letters + string.digits
        self.BA_ID_tokenizer = Tokenizer(num_words=None,
                                         filters='!"#$%()*+,./:;<=>?@[\\]^_`{|}~\t\n',
                                         lower=True,
                                         split=" ",
                                         char_level=True,
                                         oov_token=None)
        
        self.BA_ID_tokenizer.fit_on_texts(all_characters)
        
    def _encode_chars(self, words):
        return self.BA_ID_tokenizer.texts_to_sequences(words)
    
    def _encode_words(self, words):       
        return self.word_tokenizer.texts_to_sequences(words)        
        pass
    
    def _pad_sequence(self, sequence, max_len):
        return pad_sequences(sequence, max_len, padding='post')        
    
    def encode_transactions(self, df_transactions, included_columns, add_raw=False):
        
        output_data = {}
         
        for column, encoding, max_len in included_columns:
            if encoding == 'word': 
                output_data[column] = self._pad_sequence(self._encode_words(df_transactions[column]),max_len)
            elif encoding == 'char':
                output_data[column] = self._pad_sequence(self._encode_chars(df_transactions[column]),max_len)
            elif encoding == 'label':
                output_data[column] = df_transactions[column]
            
            
        if add_raw:
            output_data['raw'] = [df_transactions[x].values for x,y,z in included_columns]
            
        return output_data
    
class TransactionBatchGenerator(object):
    def __init__(self, df_transactions, TransactionConverter, included_columns):
        self.df_transactions = df_transactions
        self.TransactionConverter = TransactionConverter
        self.included_columns = included_columns
    
    def shuffle_data(self):
        self.df_transactions = self.df_transactions.sample(frac=1).reset_index(drop=True)
    
    def get_random_transactions(self, n_batch):
        random_sample = self.df_transactions.sample(n_batch)
        nn_data = self.TransactionConverter.encode_transactions(random_sample, self.included_columns)
        return nn_data
    
    def get_transactions_as_generator_object(self, n_batch, add_raw=False):
        self.num_batches = int(self.df_transactions.shape[0]/n_batch)
        for i in range(0,self.df_transactions.shape[0], n_batch):
            yield self.TransactionConverter.encode_transactions(self.df_transactions[i:i+n_batch],
                                                                self.included_columns, add_raw=add_raw)
            
    