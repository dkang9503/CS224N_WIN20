import torch
from allennlp.models import Model
import torch.nn as nn
from typing import Dict
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

class elmoModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,                 
                 out_sz: int=1):
        super().__init__()
        self.word_embeddings = word_embeddings
        self.encoder = nn.LSTM(word_embeddings.get_output_dim(), 
                               64, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss()        
        
    def forward(self, tokens: Dict[str, torch.Tensor],
                label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)        
        embeddings = self.word_embeddings(tokens)    
        
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)                

        return class_logits
    
    
def returnElmoModel():
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    
    model = elmoModel(word_embeddings)
    
    return model
