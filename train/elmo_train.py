import numpy as np
from tqdm import tqdm
import random
from datetime import date
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
import torch
import argparse
import sys
sys.path.insert(0, '../utils')
from train_utils import f_score, info
from allennlp_utils import returnDataLoader
sys.path.insert(0, '../models')
from elmo import returnElmoModel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-m', required=True, help="Which dataset to use, without .csv suffix")
parser.add_argument('--optimizer', '-opt', help="Which optimizer to use", nargs='?', type=str, default="adam")
parser.add_argument('--lr', '-lr', help="Learning Rate", nargs='?', type=int, default=2e-5)
parser.add_argument('--wd', '-wd', help="Weight Decay")
parser.add_argument('--momentum', '-mo', help="Momentum", nargs='?', type=int, default=9e-1)
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
        tokens = batch['tokens']
        tokens['tokens'] = tokens['tokens'].to(device)
        labels = batch['label'].to(device) 
    
        #Forward pass            
        outputs = model(tokens)
        loss = loss_fcn(outputs, labels)
        acc_loss += loss.item()                
        train_loss.append(loss.item())
        logits = torch.sigmoid(outputs)
        train_correct += torch.sum((logits >= .5) == labels).item()
        
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
    
    for i, batch in enumerate(valid_iter):
        tokens = batch['tokens']
        tokens['batch'] = tokens['tokens'].to(device)
        labels = batch['label'].to(device) 

        with torch.no_grad():
            outputs = model(tokens)

        loss = loss_fcn(outputs, labels)
        valid_loss.append(loss.item())
        logits = torch.sigmoid(outputs)
        predict = logits >= .5
        valid_correct += torch.sum(predict == labels).item()

        #Add to tensorboard
        writer.add_scalar('Iteration Validation Loss', loss.item(), 
                          epoch*valid_size + i + 1)

        for t, p in zip(labels, predict):
            conf_matrix[t, p] += 1            
        
    return valid_loss, valid_correct, conf_matrix
        

def train(train_iter, valid_iter, model, device):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    
    train_size = len(train_iter)
    valid_size = len(valid_iter)
    train_num_examples = len(train_iter.dataset)
    valid_num_examples = len(valid_iter.dataset)
    
    #Set model to either cpu or gpu
    model.to(device)
    
    #Define loss function
    loss_fcn = BCEWithLogitsLoss()
    
    #Set optimizers
    if args.optimizer == "adam":
        optimizer = AdamW(model.parameters(), lr = args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, 
                                     weight_decay = args.weight_decay, 
                                     momentum = args.momentum)    
        
    total_steps = len(train_iter) * args.epochs
    
    #Create linear lr scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)    
    
    #Create Tensorboard    
    today = date.today()
    date_prefix = today.strftime("%m_%d")
    log_dir_suffix = f"{date_prefix}_ELMO_{args.dataset}_lr_{args.lr}_epochs_{args.epochs}_batch_size_{args.batch_size}"
    log_dir = "../logs/elmo/" + log_dir_suffix
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
            #valid_loss, valid_correct
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
            torch.save(model.state_dict(), "../../saved_models/elmo" + log_dir_suffix + ".pth")

def main():
    print(args)
    #Load Data, probably add more here as we have more data augmentation data    
    trainLoader = returnDataLoader('../data/' + args.dataset + ".csv" , args.batch_size)
    validLoader = returnDataLoader('../data/valid.csv' , args.batch_size)
    
    #Declare model
    model = returnElmoModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train(trainLoader, validLoader, model, device)
    
if __name__ == '__main__':
    main()
    
    
