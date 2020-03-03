from eda import eda
import pandas as pd
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--da_type", required=True, type=str, help="The types of data augmentations wanted out of sr, ri, rs, rd")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
args = ap.parse_args()


def gen_eda(train_orig, output_file, da_type, alpha, num_aug=9):
    data = pd.read_csv(train_orig)
    targets = data['target'].values
    texts = data['text'].values
    
    targetsToReturn = np.zeros(num_aug*len(targets), dtype=int)
    textsToReturn = np.empty(num_aug*len(targets), dtype=object)

    for i, target in enumerate(targets):        
        sentence = texts[i]
        #Generate augmented sentences        
        aug_sentences = eda(sentence, da_type, alpha_sr=alpha, alpha_ri=alpha,\
                            alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        
        #Add them to our list
        targetsToReturn[(num_aug*i):(num_aug)*(i+1)] = [target]*num_aug
        textsToReturn[(num_aug*i):(num_aug)*(i+1)] = aug_sentences
    
    #Concatenate our list
    newTargets = np.concatenate((targets, targetsToReturn))
    newTexts = np.concatenate((texts, textsToReturn))
    
    #Create new dataframe and export into a csv
    newData = pd.DataFrame({'target': newTargets, 'text': newTexts})
    newData.to_csv(output_file, index=False)
            

if __name__ == "__main__":    
    da_type = args.da_type.split('_')
    join_char = "_"
    output = join_char.join(da_type) + "_alpha_" + str(args.alpha) \
                + "_num_aug_" + str(args.num_aug) + ".csv"
    
    gen_eda(args.input, output, da_type, alpha=args.alpha, num_aug=args.num_aug)