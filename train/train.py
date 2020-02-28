import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import argparse
import sys
from datetime import date
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, '../data')
from loading_data import createIterators

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help="Which model to use")
parser.add_argument('--dataset', '-m', help="Which dataset to use")
parser.add_argument('--optimizer', '-opt', help="Which optimizer to use", nargs='?', type=str, const="adam")
parser.add_argument('--lr', '-lr', help="Learning Rate", nargs='?', type=int, const=2e-5)
parser.add_argument('--wd', '-wd', help="Weight Decay")
parser.add_argument('--momentum', '-mo', help="Momentum", nargs='?', type=int, const=9e-1)
parser.add_argument('--step_size', '-step', help="Step size for Learning Rate decay")
parser.add_argument('--epochs', '-e', help="Number of Epochs", nargs='?', type=int, const=4)
parser.add_argument('--batch_size', '-bs', help="Batch Size", nargs='?', type=int, const=32)
parser.add_argument('--random_noise', '-e', help="Whether or not to insert random\
                                                  noise to embeddings")
args = parser.parse_args()

def train(train_iter, valid_iter, model, device):        
    
    train_size = len(train_iter.dl.dataset)
    valid_size = len(valid_iter.dl.dataset)
    
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
    
    for epoch in range(args.epochs):
        ### TRAINING ###
        train_loss = []
        train_correct = 0
        
        model.train()
        for X, y in tqdm(train_iter):            
            preds = model(X)
            loss = loss_fcn(preds, y)
            
            #Compute gradient and update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            train_correct += torch.sum((preds > .5).float() == y)
                        
        ### VALIDATION ###
        val_loss = []
        val_correct = 0
        
        model.eval()
        for batch in valid_iter:
            with torch.no_grad():
                X, y = batch
                preds = model(X)
                loss = loss_fcn(preds, y)
                
                val_loss.append(loss.item())
                val_correct += torch.sum((preds > .5).float() == y)
            
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, val_loss))
        
        ### UPDATE TENSORBOARD ###
        writer.add_scalar('Training Loss', np.mean(train_loss), epoch)
        writer.add_scalar('Training Accuracy', train_correct/train_size, epoch)
        writer.add_scalar('Validation Loss', np.mean(val_loss), epoch)
        writer.add_scalar('Validation Accuracy', val_correct/valid_size, epoch)
    
    
def main():
    #Load Data, probably add more here as we have more data augmentation data
    train_data = pd.read_csv('../data/train.csv')
    valid_data = pd.read_csv('../data/valid.csv')
    test_data = pd.read_csv('../data/test.csv')
    
    train_iter, valid_iter, test_iter= createIterators(train_data, 
                                                       valid_data, 
                                                       test_data,
                                                       batch_size = args.batch_size)
    
    #Declare model
    model = torch.nn.LSTM(5, 2) #Place holder for now
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Train our model
    train(train_iter, valid_iter, model, device)    

if __name__ == '__main__':
    main()