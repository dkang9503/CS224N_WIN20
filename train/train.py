import torch
import torch.nn as nn
import pandas as pd
import argparse
import sys
from datetime import date
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
sys.path.insert(0, '../data')
from loading_data import createIterators

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help="Which model to use")
parser.add_argument('--dataset', '-m', help="Which dataset to use")
parser.add_argument('--optimizer', '-opt', help="Which optimizer to use")
parser.add_argument('--lr', '-lr', help="Learning Rate")
parser.add_argument('--wd', '-wd', help="Weight Decay")
parser.add_argument('--momentum', '-mo', help="Momentum")
parser.add_argument('--step_size', '-mo', help="Step size for Learning Rate decay")
parser.add_argument('--epochs', '-e', help="Number of Epochs")
parser.add_argument('--batch_size', '-bs', help="Batch Size")
args = parser.parse_args()

def train(train_iterator, valid_iterator, model, device):        
    
    train_size = len(train_iterator.data())
    valid_size = len(valid_iterator.data())
    
    loss_fcn = nn.BCELoss()        
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, 
                                     weight_decay = args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, 
                                     weight_decay = args.weight_decay, 
                                     momentum = args.momentum)
    
    #Create Tensorboard    
    today = date.today()
    date_prefix = today.strftime("%m_%d")
    writer = SummaryWriter(logdir=f'../logs/{date_prefix}_{args.model}_\
                           {args.dataset}_lr_{args.lr}\
                           _wd_{args.weight_decay}_mo_{args.momentum}_\
                           steps_{args.step_size}_epochs_{args.epochs}\
                           batch_size_{args.batch_size}')
    
    model.to(device)            
    
    for epoch in args.epochs:        
        ### TRAINING ###
        train_loss = 0
        train_correct = 0
        
        model.train()
        for batch in train_iterator:
            X, y = batch
            preds = model(X)
            loss = loss_fcn(preds, y)
            
            #Compute gradient and update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()        
            train_correct += torch.sum((preds > .5).float() == y)
                        
        ### VALIDATION ###
        valid_loss = 0
        valid_correct = 0
        
        model.eval()
        for batch in valid_iterator:
            X, y = batch
            preds = model(X)
            loss = loss_fcn(preds, y)
            
            valid_loss += loss.item()
            valid_correct += torch.sum((preds > .5).float() == y)
            
        ### UPDATE TENSORBOARD ###
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Training Accuracy', train_correct/train_size, epoch)
        writer.add_scalar('Validation Loss', valid_loss, epoch)
        writer.add_scalar('Validation Accuracy', valid_correct/valid_size, epoch)
    
    
def main():
    #Load Data, probably add more here as we have more data augmentation data
    train_data = pd.read_csv('../data/train.csv')
    valid_data = pd.read_csv('../data/valid.csv')
    test_data = pd.read_csv('../data/test.csv')
    
    train_iterator, valid_iterator, test_iterator= createIterators(train_data, 
                                                                   valid_data, 
                                                                   test_data,
                                                                   batch_size = args.batch_size)
    
    #Declare model
    model = torch.nn.LSTM(5, 2) #Place holder for now
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Train our model
    train(train_iterator, valid_iterator, model, device)    

if __name__ == '__main__':
    main()