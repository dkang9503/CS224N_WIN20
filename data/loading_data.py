import torch.nn as nn
import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator
from torchtext.vocab import GloVe
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def splitAndSaveData(filepath):
    '''
        Given filepath of original dataset, will split data and return 3
        pandas dataframe objects.
    '''
    data = pd.read_csv(filepath)    
    
    X_tv, X_test, y_tv, y_test = train_test_split(data['text'], \
                                                      data['target'],
                                                      test_size=0.15,
                                                      random_state=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_tv, \
                                                      y_tv,
                                                      test_size=0.1765,
                                                      random_state=1)
    
    
    train_data = pd.DataFrame({'target' : y_train, \
                               'text' : X_train.replace('\n', '', regex = True)})
    valid_data = pd.DataFrame({'target' : y_val, \
                               'text' : X_val.replace('\n', '', regex = True)})
    test_data = pd.DataFrame({'target' : y_test, \
                              'text' : X_test.replace('\n', '', regex = True)})
    
    train_data.to_csv('train.csv', index=False)
    valid_data.to_csv('valid.csv', index=False)
    test_data.to_csv('test.csv', index=False)
    
    return train_data, valid_data, test_data


def createIterators(train_data, valid_data, test_data, batch_size=32, \
                    write=False, path="../Data/"):
    '''
        Given train/valid/test pandas dataframe objects, creates 
        iterators for all three datasets and returns them
    '''
    TEXT = Field(sequential=True)
    LABEL = Field(sequential=False, use_vocab=False)
    
    tv_datafields = [("text", TEXT), ("target", LABEL)]
    
    train, valid, test = TabularDataset.splits(
               path = ".",
               train = path+"train.csv", 
               validation = path+"valid.csv",
               test = path+"test.csv",
               format='csv',
               skip_header=True, 
               fields=tv_datafields)
    
    #Create Iterators to train over
    train_iterator = Iterator(train, sort_key = lambda x: len(x.text), batch_size=batch_size)
    valid_iterator = Iterator(valid, sort_key = lambda x: len(x.text), batch_size=batch_size)
    test_iterator = Iterator(test, sort_key = lambda x: len(x.text), batch_size=batch_size)    

    return train_iterator, valid_iterator, test_iterator

def createVocab(train_directory="train.csv", write=False):
    '''
        Downloads GloVe and returns a vocab object. GloVe is saved
        one directory below the GitHub directory called 'solver_cache'        
    '''
    
    TEXT = Field(sequential=True)
    LABEL = Field(sequential=False, use_vocab=False)
    train = TabularDataset(path='train.csv', format = 'csv', skip_header=True,
                           fields=[("text", TEXT), ("target", LABEL)])
    
    # This downloads GloVe if not already downloaded.
    glove_object = GloVe(name = 'twitter.27B', dim=100, cache="../../solver_cache")
    TEXT.build_vocab(train, vectors = glove_object)        
    
    vocab = TEXT.vocab
    
    # Initialize out of vocab word vectors
    for i in range(len(vocab)):
        if len(vocab.vectors[i,:].nonzero()) == 0:
            torch.nn.init.normal_(vocab.vectors[i])
            
    if write:
        pickle.dump(vocab, open('vocab_from_train.pkl', 'wb'))
        
    return vocab
    

if __name__ == '__main__()':

    #Split data into train/val/test from original data csv
    train_data, valid_data, test_data =splitAndSaveData('original_data_file _no_url.csv')
    
    #Simple txt file so we can do EDA
    train_data.to_csv('train.txt', sep='\t', header=False,index=False)
    
    #Create iterators with data
    train_iterator, valid_iterator, test_iterator= createIterators(train_data, valid_data, test_data)
    
    #Create vocab embedding
    vocab = createVocab(write = True) 
