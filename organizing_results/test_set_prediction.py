import glob
import sys
import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
import argparse
sys.path.insert(0, '../utils')
from allennlp_utils import returnElmoDataLoader
from bert_utils import returnBertDataloader
from loading_data import returnLSTMDataLoader
from train_utils import flat_accuracy, f_score, info
from transformers import BertForSequenceClassification
sys.path.insert(0, '../models')
from elmo import returnElmoModel
from baseline_model import baselineModel

parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-m', required=True, help="Directory of saved models")
parser.add_argument('--batch_size', '-bs', help="Batch Size", nargs='?', type=int, default=16)
args = parser.parse_args()

def pickIteratorAndModel(file, bertModel, elmoModel, device):
    if ("BERT" in file) | ("bert" in file):
        modelName = "bert"
        test_data = pd.read_csv('../data/test.csv')
        testLoader = returnBertDataloader(test_data, args.batch_size, shuffle=False)    
        model = bertModel        
    elif ("elmo" in file) | ("ELMO" in file):
        modelName = "elmo"
        testLoader = returnElmoDataLoader('../data/test.csv' , args.batch_size)
        model = elmoModel
    else:
        modelName = "lstm"
        trainLoader, _, testLoader= returnLSTMDataLoader(path = args.dataset,
                                                       batch_size = args.batch_size)
        model = baselineModel(256, trainLoader.dl.dataset.fields['text'])        
        
    model.load_state_dict(torch.load(file))
    model.to(device)
    return testLoader, model, modelName

def predict(file, test_iterator, model, modelName, device, df):
    model_dict = {}
    loss_fcn = BCEWithLogitsLoss()
    losses = []
    num_correct = 0
    conf_matrix = torch.zeros(2, 2)    
    for batch in test_iterator:
        if modelName == "bert":
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)        
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask,
                            labels=labels)            
            loss = outputs[0]
            logits = outputs[1]
            predict = torch.argmax(logits, 1)
            
        elif modelName == "elmo":
            tokens = batch['tokens']
            tokens['tokens'] = tokens['tokens'].to(device)
            labels = batch['label'].to(device)     
            outputs = model(tokens)
            loss = loss_fcn(outputs, labels)
            logits = torch.sigmoid(outputs)
            predict = (logits >= .5).flatten()
            
        else:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            loss = loss_fcn(outputs.flatten(), labels.float())
            logits = torch.sigmoid(outputs)
            predict = (logits >= .5).flatten()
           
        losses.append(loss)
        num_correct += torch.sum(predict == labels.byte()).item()
        for t, p in zip(labels, predict):
            conf_matrix[t, p] += 1
            
    J, sensitivity, specificity = info(conf_matrix)
    model_dict['loss'] = np.mean(losses)
    model_dict['name'] = file
    model_dict['acc'] = num_correct/1632
    model_dict['F1'] = f_score(conf_matrix)[0]
    model_dict['Youdens'] = J
    model_dict['sensitivity'] = sensitivity
    model_dict['specificity'] = specificity
    df = df.append([model_dict])
    
    return df
            
def main():    
    #List of models to iterate from
    fileList = glob.glob(args.directory + "/*")
    
    #Initialize dataframe to fill
    df = pd.DataFrame()
    
    bertModel = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    elmoModel = returnElmoModel()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    for file in fileList:
        test_iterator, model, modelName = pickIteratorAndModel(file, bertModel, 
                                                               elmoModel, device)
        df = predict(file, test_iterator, model, modelName, device, df)

if __name__ == '__main__':
    main()