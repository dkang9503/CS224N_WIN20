import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from datetime import date
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '../utils')
from bert_utils import returnDataloader
from train_utils import flat_accuracy, f_score, info
import torch
import argparse

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

def train(train_iter, valid_iter, model, device):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    
    train_size = len(train_iter)
    valid_size = len(valid_iter)
    train_num_examples = len(train_iter.dataset)
    valid_num_examples = len(valid_iter.dataset)
    
    #Set model to either cpu or gpu
    model.to(device)
    
    #Set optimizers
    if args.optimizer == "adam":
        optimizer = AdamW(model.parameters(), lr = args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, 
                                     weight_decay = args.weight_decay, 
                                     momentum = args.momentum)           
    
    #Create linear lr scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = train_num_examples)    
    
    #Create Tensorboard    
    today = date.today()
    date_prefix = today.strftime("%m_%d")
    log_dir_suffix = f"{date_prefix}_BERT_{args.dataset}_lr_{args.lr}_epochs_{args.epochs}_batch_size_{args.batch_size}"
    log_dir = "../logs/bert_fix/" + log_dir_suffix
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
            
            #Add to tensorboard
            writer.add_scalar('Iteration Training Loss', loss.item(), 
                              epoch*train_size + i + 1)
        
        print("Training Loss: " + str(np.mean(train_loss)) + \
              ", Training Accuracy : " + str(train_correct/train_num_examples))
       
        ### VALIDATION ###        
        valid_loss = []
        valid_correct = 0

        conf_matrix = torch.zeros(2, 2)
        
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
            predict = torch.argmax(logits, 1)
            valid_correct += torch.sum(predict == labels).item()

            #Add to tensorboard
            writer.add_scalar('Iteration Validation Loss', loss.item(), 
                              epoch*valid_size + i + 1)

            for t, p in zip(labels, predict):
                conf_matrix[t, p] += 1



        print("Validation Loss: " + str(np.mean(valid_loss)) + \
              ", Validation Accuracy : " + str(valid_correct/valid_num_examples))

        J, sensitivity, specificity = info(conf_matrix)
               
        ### UPDATE TENSORBOARD ###
        print(epoch)
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
        '''
        if np.mean(valid_loss) < best_loss:
            best_loss = np.mean(valid_loss)
            torch.save(model.state_dict(), "../../saved_models/" + log_dir_suffix + ".pth")
        '''

def main():
    print(args)
    #Load Data, probably add more here as we have more data augmentation data
    train_data = pd.read_csv('../data/' + args.dataset + ".csv")
    valid_data = pd.read_csv('../data/valid.csv')    
    
    trainLoader = returnDataloader(train_data, args.batch_size)
    validLoader = returnDataloader(valid_data, args.batch_size)
    
    #Declare model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train(trainLoader, validLoader, model, device)

    # calculate F score
    
if __name__ == '__main__':
    main()
    
    
