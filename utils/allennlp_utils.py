import torch
from allennlp.data.instance import Instance
from typing import List, Dict, Union, Callable, Optional, Iterator
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from overrides import overrides
from allennlp_batch import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]

def tokenizer(x: str):
    return [w.text for w in
            SpacyWordSplitter(language='en_core_web_sm', 
                              pos_tags=False).split_words(x)[:50]]

def allennlp_collate(instances: List[Instance]) -> TensorDict:
    batch = Batch(instances)
    return batch.as_tensor_dict(batch.get_padding_lengths())

#my_loader = DataLoader(train_ds, batch_size=16, collate_fn=allennlp_collate)
    
class TweetDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int]=50) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len

    @overrides
    def text_to_instance(self, tokens: List[Token],
                         labels: np.ndarray=None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}                                
        label_field = ArrayField(labels)
        fields["label"] = label_field

        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)        
        for i, row in df.iterrows():            
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["text"])],
                np.array([row["target"]]),
            )

def returnDataLoader(filepath, batch_size):
    token_indexer = ELMoTokenCharactersIndexer()
    reader = TweetDatasetReader(tokenizer=tokenizer, 
                                token_indexers={"tokens": token_indexer})
    iterator = BucketIterator(batch_size=batch_size, 
                              sorting_keys=[("tokens", "num_tokens")],
                              track_epoch= True)
    vocab = Vocabulary()
    iterator.index_with(vocab)
    # First pass through data to make sure vocab loaded for the dataset
    dataset = reader.read(filepath)
    filler = iterator(dataset, num_epochs = 1)
    for batch in filler:
        pass
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=allennlp_collate, shuffle = True)    
    
    return dataloader
        
    
                      
    
    