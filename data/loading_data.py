import torch.nn as nn
import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def splitAndSaveData(filepath):
    data = pd.read_csv(filepath)    
    
    X_tv, X_test, y_tv, y_test = train_test_split(data['text'], \
                                                      data['target'],
                                                      test_size=0.3,
                                                      random_state=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_tv, \
                                                      y_tv,
                                                      test_size=0.2,
                                                      random_state=1)
    
    
    train_data = pd.DataFrame({'text' : X_train, 'target' : y_train})
    valid_data = pd.DataFrame({'text' : X_train, 'target' : y_train})
    test_data = pd.DataFrame({'text' : X_train, 'target' : y_train})
    
    train_data.to_csv('train.csv')
    valid_data.to_csv('valid.csv')
    test_data.to_csv('test.csv')
    
    return train_data, valid_data, test_data


def createIterators(train_data, valid_data, test_data, batch_size=32, write=False):
    TEXT = Field(sequential=True)
    LABEL = Field(sequential=False, use_vocab=False)
    
    tv_datafields = [("text", TEXT), ("target", LABEL)]
    
    train, valid, test = TabularDataset.splits(
               train="train.csv", 
               validation="valid.csv",
               test="test.csv",
               format='csv',
               skip_header=True, 
               fields=tv_datafields)
    
    TEXT.build_vocab(train, vectors = 'glove.twitter.27B.100d')
    
    #Create Iterators to train over
    train_iterator = Iterator(train, sort_key = lambda x: len(x.text), batch_size=batch_size)
    valid_iterator = Iterator(valid, sort_key = lambda x: len(x.text), batch_size=batch_size)
    test_iterator = Iterator(test, sort_key = lambda x: len(x.text), batch_size=batch_size)
    
    vocab = TEXT.vocab
    
    # Initialize out of vocab word vectors
    for i in range(len(vocab)):
        if len(vocab.vectors[i,:].nonzero()) == 0:
            torch.nn.init.normal_(vocab.vectors[i])
            
    if write:
        pickle.dump(vocab, open('vocab_from_train.pkl', 'wb'))
        pickle.dump(train_iterator, open('train_iterator.pkl', 'wb'))
        pickle.dump(valid_iterator, open('valid_iterator.pkl', 'wb'))
        pickle.dump(test_iterator, open('test_iterator.pkl', 'wb'))

    return train_iterator, valid_iterator, test_iterator, vocab
    

if __name__ == '__main__()':

    #Split data into train/val/test from original data csv
    train_data, valid_data, test_data =splitAndSaveData('original_data_file.csv')
    
    #Create word embeddings and iterators with data
    train_iterator, valid_iterator, test_iterator, vocab = createIterators(train_data, valid_data, test_data)
