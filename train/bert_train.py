import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from datetime import date
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '../utils')
from bert_utils import tokenize, make_mask
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-m', required=True, help="Which dataset to use, without .csv suffix")
parser.add_argument('--optimizer', '-opt', help="Which optimizer to use", nargs='?', type=str, default="adam")
parser.add_argument('--lr', '-lr', help="Learning Rate", nargs='?', type=int, default=2e-5)
parser.add_argument('--wd', '-wd', help="Weight Decay")
parser.add_argument('--momentum', '-mo', help="Momentum", nargs='?', type=int, default=9e-1)
parser.add_argument('--step_size', '-step', help="Step size for Learning Rate decay")
parser.add_argument('--epochs', '-e', help="Number of Epochs", nargs='?', type=int, default=6)
parser.add_argument('--batch_size', '-bs', help="Batch Size", nargs='?', type=int, default=16)
args = parser.parse_args()

def train(train_iter, valid_iter, model, device):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    
    train_size = len(train_iter)
    valid_size = len(valid_iter)
    
    #Set model to either cpu or gpu
    model.to(device)
    
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
    log_dir = f"../logs/{date_prefix}_BERT_{args.dataset}_lr_{args.lr}_epochs_{args.epochs}_batch_size_{args.batch_size}"
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
          
    model.zero_grad()
    
    for epoch in range(args.epochs):
        ### TRAINING ###
        print("Beginning epoch " + str(epoch))
        train_loss = []
        train_correct = 0
        
        model.train() #Set train mode
        
        for i, batch in enumerate(tqdm(train_iter)):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)        

            #Forward pass            
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask,
                        labels=labels)            
            loss = outputs[0]            
            train_loss.append(loss.item())
            logits = outputs[1]
            train_correct += torch.sum(torch.argmax(logits, 1) == labels).item()
            
            #Compute gradient and update params
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip gradient
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            writer.add_scalar('Iteration Training Loss', loss.item(), 
                              epoch*train_size + i + 1)
        
        print("Training Loss: " + str(np.mean(train_loss)) + \
              ", Training Accuracy : " + str(train_correct/(train_size * args.batch_size)))
       
        ### VALIDATION ###        
        valid_loss = []
        valid_correct = 0
        
        model.eval()
        for i, batch in enumerate(valid_iter):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)        
            
            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None, 
                                attention_mask=input_mask, labels=labels)            
            
            loss = outputs[0]
            valid_loss.append(loss.item())
            logits = outputs[1]
            valid_correct += torch.sum(torch.argmax(logits, 1) == labels).item()                       
            
        print("Validation Loss: " + str(np.mean(valid_loss)) + \
              ", Validation Accuracy : " + str(valid_correct/(valid_size * args.batch_size)))
              
        ### UPDATE TENSORBOARD ###
        writer.add_scalar('Epoch Training Loss', np.mean(train_loss), epoch)
        writer.add_scalar('Epoch Validation Loss', np.mean(valid_loss), epoch)
        writer.add_scalar('Epoch Training Accuracy', 
                          train_correct/(train_size * args.batch_size), epoch)
        writer.add_scalar('Epoch Validation Accuracy', 
                          valid_correct/(valid_size * args.batch_size), epoch)

def returnDataloader(data, batch_size):
    sentences = data.text.values
    labels = data.target.values
    
    # Tokenize all of the sentences and map the tokens to their word IDs.
    tokens = tokenize(sentences)
    
    # Create attention masks
    masks = make_mask(tokens)
    
    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    tokens  = torch.tensor(tokens)  
    labels = torch.tensor(labels)   
    masks  = torch.tensor(masks)
    
    # Create the DataLoader
    dataset = TensorDataset(tokens, masks, labels)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return dataloader

def main():
    print(args)
    #Load Data, probably add more here as we have more data augmentation data
    train_data = pd.read_csv('../data/' + args.dataset + ".csv")
    valid_data = pd.read_csv('../data/valid.csv')    
    
    trainLoader = returnDataloader(train_data, args.batch_size)
    validLoader = returnDataloader(valid_data, args.batch_size)
    
    #Declare model
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-cased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train(trainLoader, validLoader, model, device)
    
if __name__ == '__main__':
    main()
    
    