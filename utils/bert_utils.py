import numpy as np
import torch
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case = False)

# Tokenizes sentences for BERT. Argument is a pandas dataframe
def tokenize(sentences):
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Add padding
        #   (6) Return pytorch tensors
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length = 50, # Maximum length of a tweet is 280, temporarily set lower because my local GPU does not have enough memory
            pad_to_max_length = True, # Add padding
            #return_tensors = 'pt' #Returng PyTorch tensors
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    return input_ids

# Creates attention masks. A requirement for this implementation of BERT.
# Argument is a list of tokenized sentences
def make_mask(sentences):
    attention_masks = []

    # For each sentence...
    for sent in sentences:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return attention_masks

#Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def returnDataloader(data, batch_size, shuffle=True):
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
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader