import torch
import numpy as np
import random
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import pandas as pd
import argparse
from datetime import date
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '../utils')
from train_utils import f_score, info
sys.path.insert(0, '../data')
from loading_data import createIterators
sys.path.insert(0, '../models')
from baseline_model import baselineModel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-m', required=True, help="Which dataset to use, without .csv suffix")
parser.add_argument('--optimizer', '-opt', help="Which optimizer to use", nargs='?', type=str, default="adam")
parser.add_argument('--lr', '-lr', help="Learning Rate", nargs='?', type=float, default=3e-3)
parser.add_argument('--wd', '-wd', help="Weight Decay", nargs='?', type=float, default=0)
parser.add_argument('--momentum', '-mo', help="Momentum", nargs='?', type=float, default=9e-1)
parser.add_argument('--step_size', '-step', help="Step size for Learning Rate decay")
parser.add_argument('--epochs', '-e', help="Number of Epochs", nargs='?', type=int, default=5)
parser.add_argument('--batch_size', '-bs', help="Batch Size", nargs='?', type=int, default=16)
args = parser.parse_args()


def train_one_epoch(epoch, train_size, model, device, train_loader, epoch_pbar, 
                    optimizer, scheduler, writer, loss_fcn):
    train_loss = []
    train_correct = 0
    acc_loss =0
    acc_avg = 0    
    for i, batch in enumerate(train_loader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
    
        #Forward pass            
        outputs = model(inputs)
        loss = loss_fcn(outputs.flatten(), labels.float())
        acc_loss += loss.item()                
        train_loss.append(loss.item())
        logits = torch.sigmoid(outputs)
        train_correct += torch.sum((logits >= .5).flatten() == labels.byte()).item()
        
        #Update progress bar
        avg_loss = acc_loss/(i + 1)                
        acc_avg = train_correct/((i+1) * 16)
        desc = f"Epoch {epoch} - loss {avg_loss:.4f} - acc {acc_avg:.4f} - lr {optimizer.param_groups[0]['lr']}"
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)
        
        #Compute gradient and update params
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip gradient
        optimizer.step()                
        scheduler.step()
        optimizer.zero_grad()
        model.zero_grad()
        
        #Write to Tensorboard
        writer.add_scalar('Iteration Training Loss', loss.item(), 
                              epoch*train_size + i + 1)
    
    return train_loss, train_correct

def valid_one_epoch(epoch, valid_size, model, device, valid_iter, epoch_pbar, 
                    optimizer, scheduler, writer, loss_fcn, conf_matrix):
    valid_loss = []
    valid_correct = 0
    acc_loss = 0
    acc_avg = 0
        
    for i, batch in enumerate(valid_iter):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        loss = loss_fcn(outputs.flatten(), labels.float())
        valid_loss.append(loss.item())
        acc_loss += loss.item()                
        logits = torch.sigmoid(outputs)
        predict = (logits >= .5).flatten()
        valid_correct += torch.sum(predict == labels.byte()).item()
        
        #Update progress bar
        avg_loss = acc_loss/(i + 1)                
        acc_avg = valid_correct/((i+1) * 16)
        desc = f"Epoch {epoch} - loss {avg_loss:.4f} - acc {acc_avg:.4f}"
        epoch_pbar.set_description(desc)
        epoch_pbar.update(1)

        #Add to tensorboard
        writer.add_scalar('Iteration Validation Loss', loss.item(), 
                          epoch*valid_size + i + 1)

        #Conver to indices
        labels = labels.long()
        predict = predict.long()
        for t, p in zip(labels, predict):
            conf_matrix[t, p] += 1            
        
    return valid_loss, valid_correct, conf_matrix

def train(train_iter, valid_iter, model, device):        
    
    train_size = len(train_iter)
    valid_size = len(valid_iter)
    train_num_examples = len(train_iter.dl.dataset)
    valid_num_examples = len(valid_iter.dl.dataset)
    
    #Set model to either cpu or gpu
    model.to(device)            
    
    #Define loss function
    loss_fcn = BCEWithLogitsLoss()
    
    #Set optimizers
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, 
                                     weight_decay = args.wd)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, 
                                     weight_decay = args.wd, 
                                     momentum = args.momentum)
    
    #Create linear lr scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = train_num_examples)    
    
    #Create Tensorboard    
    today = date.today()
    date_prefix = today.strftime("%m_%d")
    log_dir_suffix = f"{date_prefix}_baseline_{args.dataset}_lr_{args.lr}_epochs_{args.epochs}_batch_size_{args.batch_size}"
    log_dir = "../logs/baseline/" + log_dir_suffix
    writer = SummaryWriter(log_dir=log_dir)        
    
    best_loss = 1e9
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
          
    model.zero_grad()    
    
    for epoch in range(args.epochs):
        ### TRAINING ###
        print("Beginning Training in Epoch " + str(epoch))
        with tqdm(total = train_size) as epoch_pbar:
            model.train()
            train_loss, train_correct = train_one_epoch(epoch, train_size, model, 
                                            device, train_iter, epoch_pbar, 
                                            optimizer, scheduler, writer, loss_fcn)                    
                        
        ### VALIDATION ###
        print("Beginning Validation in Epoch " + str(epoch))
        valid_loss = []
        valid_correct = 0

        conf_matrix = torch.zeros(2, 2)
                
        with tqdm(total = valid_size) as epoch_pbar:
            model.eval()                           
            valid_loss, valid_correct, conf_matrix = valid_one_epoch(epoch, valid_size, model, 
                                                                     device, valid_iter, epoch_pbar, 
                                                                     optimizer, scheduler, writer, 
                                                                     loss_fcn, conf_matrix)                                    

        J, sensitivity, specificity = info(conf_matrix)
               
        ### UPDATE TENSORBOARD ###
        writer.add_scalar('Epoch Training Loss', np.mean(train_loss), epoch)
        writer.add_scalar('Epoch Validation Loss', np.mean(valid_loss), epoch)
        writer.add_scalar('Epoch Training Accuracy', 
                          train_correct/train_num_examples, epoch)
        writer.add_scalar('Epoch Validation Accuracy', 
                          valid_correct/valid_num_examples, epoch)
        writer.add_scalar('F1 Score', f_score(conf_matrix)[0], epoch)
        writer.add_scalar('Youdens', J, epoch)
        writer.add_scalar('Sensitivity', sensitivity, epoch)
        writer.add_scalar('Specificity', specificity, epoch)

        ### Save if Model gets best loss ###
        
        if np.mean(valid_loss) < best_loss:
            best_loss = np.mean(valid_loss)
            torch.save(model.state_dict(), "../../saved_models/baseline/" + log_dir_suffix + ".pth")
        
def main():    
    print(args)    
    
    #Load Data, probably add more here as we have more data augmentation data
    train_data = pd.read_csv(f'../data/{args.dataset}.csv')
    valid_data = pd.read_csv('../data/valid.csv')
    test_data = pd.read_csv('../data/test.csv')
    
    train_iter, valid_iter, test_iter= createIterators(train_data, 
                                                       valid_data, 
                                                       test_data,
                                                       path = args.dataset,
                                                       batch_size = args.batch_size)
    
    #Declare model and load GloVe
    model = baselineModel(256, train_iter.dl.dataset.fields['text'])    
    model.embedding.weight = torch.nn.Parameter(train_iter.dl.dataset.fields['text'].vocab.vectors)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    #Train our model        
    train(train_iter, valid_iter, model, device)    

if __name__ == '__main__':
    main()