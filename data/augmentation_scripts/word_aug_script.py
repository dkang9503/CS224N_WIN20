################################################################################
# Imports
################################################################################
import os
import re
import pandas as pd
import numpy as np
import argparse
import nlpaug.augmenter.word as naw
from tqdm import tqdm

np.random.seed(1)

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']

################################################################################
# Arguments
################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str,
                help="input file of unaugmented data")
ap.add_argument("--da_type", required=True, type=str,
                help="The types of data augmentations wanted out of " +
                     "{w2v, glovebase, glovetwit, contextin, contextsub}")
ap.add_argument("--num_aug", required=False, type=int,
                help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float,
                help="percent of words in each sentence to be changed")
ap.add_argument("--output_dir", default='./', required=False, type=str,
                help="directory to save output")
ap.add_argument("--model_dir", default='./', required=False, type=str,
                help="directory to saved models for augmentation")
args = ap.parse_args()


################################################################################
# Helper Functions
################################################################################
def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm 1234567890#':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


################################################################################
# Augmentation Functions
################################################################################
# word2vec similarity augmentation
def w2v_aug(sentence, model_path, a, n):
    aug = naw.WordEmbsAug(model_type='word2vec', model_path=model_path,
                          action='substitute' , aug_p=a, stopwords=stop_words)
    aug_sentences = aug.augment(sentence, n=n)
    if type(aug_sentences) == list:
        return aug_sentences
    else:
        return [aug_sentences]


# glove similarity augmentation
def glove_aug(sentence, model_path, a, n):
    aug = naw.WordEmbsAug(model_type='glove', model_path=model_path,
                          action='substitute', aug_p=a, stopwords=stop_words)
    aug_sentences = aug.augment(sentence, n=n)
    if type(aug_sentences) == list:
        return aug_sentences
    else:
        return [aug_sentences]


# contextual sentence augmentation
def context_aug(sentence, action, a, n, model='bert-base-uncased'):
    aug = naw.ContextualWordEmbsAug(model_path=model, action=action, aug_p=a,
                                    stopwords=stop_words)
    aug_sentences = aug.augment(sentence, n=n)
    if type(aug_sentences) == list:
        return aug_sentences
    else:
        return [aug_sentences]


################################################################################
# Top Level Augmentation Functions
################################################################################
# Augment a single example
def augment(sentence, da, alpha, num_aug, model_dir):
    sentence = get_only_chars(sentence)
    augmented_sentences = []
    num_new_per_technique = int(np.ceil(num_aug / len(da)))

    # directories
    w2v_path = model_dir + '/word2vec/GoogleNews-vectors-negative300.bin'
    glove_path = model_dir + '/GloVe/glove.6B/glove.6B.300d.txt'
    glove_path_twitter = model_dir + '/GloVe/glove.twitter.27B/glove.twitter.27B.200d.txt'

    # Word Level
    if 'w2v' in da:
        augmented_sentences += w2v_aug(sentence, model_path=w2v_path,
                                       a=alpha, n=num_new_per_technique)
    if 'glovetwit' in da:
        augmented_sentences += glove_aug(sentence, model_path=glove_path_twitter,
                                         a=alpha, n=num_new_per_technique)
    if 'glovebase' in da:
        augmented_sentences += glove_aug(sentence, model_path=glove_path,
                                         a=alpha, n=num_new_per_technique)
    if 'contextin' in da:
        augmented_sentences += context_aug(sentence, action='insert',
                                           a=alpha, n=num_new_per_technique)
    if 'contextsub' in da:
        augmented_sentences += context_aug(sentence, action='insert',
                                           a=alpha, n=num_new_per_technique)

    # Remove excess examples
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    np.random.shuffle(augmented_sentences)
    if len(da) >= 1:
        augmented_sentences = augmented_sentences[:num_aug]

    # Return
    return augmented_sentences


# Augmenta all data
def gen_nlpaug(train_orig, output_file, model_dir, da, alpha=0.1, num_aug=5):
    # get data
    data = pd.read_csv(train_orig)
    targets = data['target'].values
    texts = data['text'].values
    # initialize output data
    new_targets = np.zeros(num_aug * len(targets), dtype=int)
    new_texts = np.empty(num_aug * len(targets), dtype=object)

    for i, target in enumerate(tqdm(targets)):
        sentence = texts[i]
        # AUGMENT
        aug_sentences = augment(sentence, da, alpha=alpha,
                                num_aug=num_aug, model_dir=model_dir)

        # Add to output
        new_targets[(num_aug * i):num_aug * (i + 1)] = [target] * num_aug
        new_texts[(num_aug * i):num_aug * (i + 1)] = aug_sentences

    # Concatenate our list
    output_targets = np.concatenate((targets, new_targets))
    output_texts = np.concatenate((texts, new_texts))

    # Create new dataframe and export into a csv
    new_data = pd.DataFrame({'target': output_targets, 'text': output_texts})
    new_data.to_csv(output_file, index=False)


################################################################################
# Main
################################################################################
if __name__ == "__main__":
    da_type = args.da_type.split('_')
    join_char = "_"
    output = str(args.output_dir) + join_char.join(da_type) + "_alpha_" + str(args.alpha)
    output += "_num_aug_" + str(args.num_aug) + ".csv"

    gen_nlpaug(args.input, output, args.model_dir, da_type, alpha=args.alpha,
               num_aug=args.num_aug)
