from googletrans import Translator
import pandas as pd
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data", nargs='?', const='train.csv')
ap.add_argument("--lang", required=True, type=str, help="The intermediary language for backtranslation", nargs='?', const='spanish')
args = ap.parse_args()

if __name__ == "__main__":
    #Load data
    data = pd.read_csv(args.input)
    targets = data['target'].values
    texts = data['text'].values

    #Initiate Translator
    translator = Translator()

    #Initialize vector for new translations            
    textsToReturn = np.empty(len(targets), dtype=object)
    
    print("Currently translating in " + args.lang)
    #Perform translations
    for i in range(len(data)):
        if i % 500 == 0:
            print("We are currently on iteration " + str(i) + " out of " + str(len(data)))
        example = texts[i]
        forward_translation = translator.translate(example, dest=args.lang, src='english')
        backward_translation = translator.translate(forward_translation.text, dest='english', src=args.lang)
        
        textsToReturn[i] = backward_translation.text
    
    print("Translation complete")
    #Create new Dataframe and save it
    newTargets = np.concatenate((targets, targets))
    newTexts = np.concatenate((texts, textsToReturn))
    
    newData = pd.DataFrame({'target': newTargets, 'text': newTexts})
    newData.to_csv("backtranslation_"+args.lang+".csv", index=False)