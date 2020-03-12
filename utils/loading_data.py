import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator
from torchtext.data import BucketIterator
from torchtext.vocab import GloVe
import pandas as pd
import re
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

def splitAndSaveData(filepath, conf=False):
    '''
        Given filepath of original dataset, will split data and return 3
        pandas dataframe objects. Split is 70-15-15 train-valid-test
    '''
    data = pd.read_csv(filepath)    
        
    tv, test = train_test_split(data, test_size=0.15, random_state=1)
    train, valid = train_test_split(tv, test_size=0.1765, random_state=1)
    
    train_data = pd.DataFrame({'target' : train['target'].values, \
                               'text' : cleanData(train['text'].values),
                               'conf': train['choose_one:confidence']})
    valid_data = pd.DataFrame({'target' : valid['target'].values, \
                               'text' : cleanData(valid['text'].values),
                               'conf': valid['choose_one:confidence']})
    test_data = pd.DataFrame({'target' : test['target'].values, \
                              'text' : cleanData(test['text'].values),
                              'conf': test['choose_one:confidence']})
    if conf:        
        train_data.to_csv('../data/train_conf.csv', index=False)
        valid_data.to_csv('../data/valid_conf.csv', index=False)
        test_data.to_csv('../data/test_conf.csv', index=False)
    else:
        train_data = train_data[['target', 'text']]
        valid_data = valid_data[['target', 'text']]
        test_data = test_data[['target', 'text']]
        
        train_data.to_csv('../data/train.csv', index=False)
        valid_data.to_csv('../data/valid.csv', index=False)
        test_data.to_csv('../data/test.csv', index=False)
    
    return train_data, valid_data, test_data

def cleanData(listOfText):
        
    toReturn = []
    
    for line in listOfText:        
        clean_line = ""
        line = line.replace("’", "")
        line = line.replace("'", "")
        line = line.replace("-", " ") #replace hyphens with spaces
        line = line.replace("\t", " ")
        line = line.replace("\n", " ")
        line = re.sub('\\\\[a-zA-z0-9]+', '', line)
        line = line.lower()
        for char in line:
            if char in 'qwertyuiopasdfghjklzxcvbnm 1234567890#,@':
                clean_line += char
            elif char == '.':
                clean_line+= ''
            else:
                clean_line += ' '
        
        clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
        if clean_line[-1] == ' ':
            clean_line = clean_line[:-1]
            
        toReturn.append(clean_line)
        
    return toReturn
        

def returnLSTMDataLoader(path="train", batch_size=16, write=False):
    '''
        Given train/valid/test pandas dataframe objects, creates 
        iterators for all three datasets and returns them
    '''
    TEXT = Field(sequential=True)
    LABEL = Field(sequential=False, use_vocab=False)
    
    tv_datafields = [("target", LABEL), ("text", TEXT)]
    
    train, valid, test = TabularDataset.splits(
               path = ".",
               train = f"../data/{path}.csv", 
               validation = "../data/valid.csv",
               test = "../data/test.csv",
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
    train_data, valid_data, test_data =splitAndSaveData('../data/original_data_file_no_url.csv', True)    
    
    #Create iterators with data
    train_iter, valid_iter, test_iter = returnLSTMDataLoader(train_data, valid_data, test_data)    
    
    #run this line to see how the batch iterator works
    # next(train_iter.__iter__())