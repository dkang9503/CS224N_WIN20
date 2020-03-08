################################################################################
# Imports
################################################################################
import os
import re
import pandas as pd
import numpy as np
import argparse
import nlpaug.augmenter.sentence as nas
from tqdm import tqdm


################################################################################
# Parameters
################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int,
                help="number of augmented sentences per original sentence")
ap.add_argument("--output_dir", default='./', required=False, type=str,
                help="directory to save output")
ap.add_argument("--model_type", default='gpt2', required=False, type=str,
                help="model used for generation from" +
                "{gpt2, distilgpt2, xlnet-base-cased")
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
# Sentence-Level Augmentation Functions
################################################################################
# Generates a new sentence based on the original
def sentence_aug(sentence, n, model='gpt2'):
    aug = nas.ContextualWordEmbsForSentenceAug(model_path=model)
    aug_sentences = aug.augment(sentence, n=n)
    if type(aug_sentences) == list:
        return [get_only_chars(s) for s in aug_sentences]
    else:
        return [aug_sentences]


################################################################################
# Top Level Augmentation Functions
################################################################################
# Augment a single example
def augment(sentence, num_aug, gen_model):
    sentence = get_only_chars(sentence)
    augmented_sentences = sentence_aug(sentence, n=num_aug, model=gen_model)

    # Remove excess examples
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    np.random.shuffle(augmented_sentences)

    # Return
    return augmented_sentences


# Augmenta all data
def gen_nlpaug(train_orig, output_file, gen_model, num_aug):
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
        aug_sentences = augment(sentence, num_aug=num_aug, gen_model=gen_model)

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
    output = "sentence_gen_num_aug_" + str(args.num_aug)
    output += "_model_" + str(args.model_type) + ".csv"

    gen_nlpaug(args.input, output, gen_model=args.model_type, num_aug=args.num_aug)
