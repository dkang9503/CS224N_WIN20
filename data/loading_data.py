import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator
from torchtext.data import BucketIterator
from torchtext.vocab import GloVe
import pandas as pd
from sklearn.model_selection import train_test_split

class BatchWrapper:
    def __init__(self, dl, x_var, y_var):
        self.dl, self.x_var, self.y_var = dl, x_var, y_var # we pass in the list of attributes for x 
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) # Retrieves text for batch
            y = getattr(batch, self.y_var) # Retrieves label for batch
            yield (x, y)
        
    def __len__(self):
        return len(self.dl)   

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
    
    tv_datafields = [("target", LABEL), ("text", TEXT)]
    
    train, valid, test = TabularDataset.splits(
               path = ".",
               train = path+"train.csv", 
               validation = path+"valid.csv",
               test = path+"test.csv",
               format='csv',
               skip_header=True, 
               fields=tv_datafields)
    
    # This downloads GloVe if not already downloaded.
    glove_object = GloVe(name = 'twitter.27B', dim=100, cache="../../solver_cache")
    TEXT.build_vocab(train, vectors = glove_object)
    
    # Initialize out of vocab word vectors
    torch.manual_seed(1)
    for i in range(len(TEXT.vocab)):
        if len(TEXT.vocab.vectors[i,:].nonzero()) == 0:
            torch.nn.init.normal_(TEXT.vocab.vectors[i])
    
    #Create Iterators to train over
    train_iter, valid_iter = BucketIterator.splits((train, valid), \
                                                   batch_sizes=(batch_size,batch_size),\
                                                   sort_key=lambda x: len(x.text),\
                                                   sort_within_batch=False, \
                                                   repeat=False)                                                  
    test_iter = Iterator(test, sort = False, batch_size=batch_size, \
                             sort_within_batch=False, repeat=False)    
    
    train_dl = BatchWrapper(train_iter, "text", "target")
    valid_dl = BatchWrapper(valid_iter, "text", "target")
    test_dl = BatchWrapper(test_iter, "text", "target")

    return train_dl, valid_dl, test_dl

if __name__ == '__main__':
    #Split data into train/val/test from original data csv
    train_data, valid_data, test_data =splitAndSaveData('original_data_file _no_url.csv')
    
    #Simple txt file so we can do EDA
    train_data.to_csv('train.txt', sep='\t', header=False,index=False)
    
    #Create iterators with data
    train_iter, valid_iter, test_iter = createIterators(train_data, valid_data, test_data)    