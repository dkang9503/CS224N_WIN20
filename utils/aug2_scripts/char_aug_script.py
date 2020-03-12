################################################################################
# Imports
################################################################################
import os
import re
import pandas as pd
import numpy as np
import argparse
import nlpaug.augmenter.char as nac
from tqdm import tqdm


################################################################################
# Parameters
################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str,
                help="input file of unaugmented data")
ap.add_argument("--da_type", required=True, type=str,
                help="The types of data augmentations wanted out of " +
                     "{ocr, key, randin, randsub, randswap, randdel}")
ap.add_argument("--num_aug", default=1, required=False, type=int,
                help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", default=.25, required=False, type=float,
                help="percent of words in each sentence to be changed")
ap.add_argument("--alpha_char", default=.25, required=False, type=float,
                help="percent of characters in each word to be changed")
ap.add_argument("--output_dir", default='./', required=False, type=str,
                help="directory to save output")
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
# Character-Level Augmentation Functions
################################################################################
# Optical character augmentation
def ocr_aug(sentence, a_c,
            a_w, n):
    aug = nac.OcrAug(aug_char_p=a_c, aug_word_p=a_w)
    aug_sentences = aug.augment(sentence, n=n)
    if type(aug_sentences) == list:
        return aug_sentences
    else:
        return [aug_sentences]


# Keyboard distance augmentation
def key_aug(sentence, a_c,
            a_w, n):
    aug = nac.KeyboardAug(aug_char_p=a_c, aug_word_p=a_w)
    aug_sentences = aug.augment(sentence, n=n)
    if type(aug_sentences) == list:
        return aug_sentences
    else:
        return [aug_sentences]


# Random augmentation
def rand_aug(sentence, action, a_c,
             a_w, n):
    aug = nac.RandomCharAug(action=action, aug_char_p=a_c, aug_word_p=a_w)
    aug_sentences = aug.augment(sentence, n=n)
    if type(aug_sentences) == list:
        return aug_sentences
    else:
        return [aug_sentences]


################################################################################
# Top Level Augmentation Functions
################################################################################
# Augment a single example
def augment(sentence, da, alpha, num_aug, alpha_char):
    sentence = get_only_chars(sentence)
    augmented_sentences = []
    num_new_per_technique = int(np.ceil(num_aug / len(da)))

    # Character Level
    if 'ocr' in da:
        augmented_sentences += ocr_aug(sentence, a_c=alpha_char,
                                       a_w=alpha, n=num_new_per_technique)
    if 'key' in da:
        augmented_sentences += key_aug(sentence, a_c=alpha_char,
                                       a_w=alpha, n=num_new_per_technique)
    if 'rand_in' in da:
        augmented_sentences += rand_aug(sentence, 'insert', a_c=alpha_char,
                                        a_w=alpha, n=num_new_per_technique)
    if 'rand_sub' in da:
        augmented_sentences += rand_aug(sentence, 'sub', a_c=alpha_char,
                                        a_w=alpha, n=num_new_per_technique)
    if 'rand_swap' in da:
        augmented_sentences += rand_aug(sentence, 'swap', a_c=alpha_char,
                                        a_w=alpha, n=num_new_per_technique)
    if 'rand_del' in da:
        augmented_sentences += rand_aug(sentence, 'delete', a_c=alpha_char,
                                        a_w=alpha, n=num_new_per_technique)

    # Remove excess examples
    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    np.random.shuffle(augmented_sentences)
    if len(da) >= 1:
        augmented_sentences = augmented_sentences[:num_aug]

    if len(augmented_sentences) < num_aug:
        augmented_sentences += [sentence]*(num_aug-len(augmented_sentences))

    # Return
    return augmented_sentences


# Augmenta all data
def gen_nlpaug(train_orig, output_file, da_type, alpha=0.1, alpha_char=0.25, num_aug=5):
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
        aug_sentences = augment(sentence, da_type,
                                alpha=alpha, num_aug=num_aug, alpha_char=alpha_char)

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
    output += "_alpha_char_" + str(args.alpha_char)
    output += "_num_aug_" + str(args.num_aug) + ".csv"

    gen_nlpaug(args.input, output, da_type, alpha=args.alpha,
               alpha_char=args.alpha_char, num_aug=args.num_aug)
