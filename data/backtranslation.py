from google.cloud import translate_v2 as translate
import pandas as pd
import time
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data", nargs='?', const='train.csv')
ap.add_argument("--lang", required=True, type=str, help="The intermediary language for backtranslation", nargs='?', const='es')
args = ap.parse_args()

if __name__ == "__main__":
    #Load data
    data = pd.read_csv(args.input)
    targets = data['target'].values
    texts = data['text'].values
    
    #Initiate Translator
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcloud.json'
    translate_client = translate.Client()    

    #Initialize vector for new translations            
    textsToReturn = np.empty(len(targets), dtype=object)
    
    print("Currently translating in " + args.lang)
    
    #Due to limitations of google API, process in batches
    divide_length = 100
    iter_limit = int(len(texts) / divide_length) + 1
    
    for i in range(iter_limit):
        print("Iteration " + str(i) + " out of " + str(iter_limit))
        #Get current batch of tweets
        batch = texts[(divide_length*i):(divide_length*(i+1))]
        
        #Translate to target language
        to_trans = translate_client.translate(list(batch), target_language=args.lang,
                                                             source_language='en')
        to_trans_list = [translation['translatedText'] for translation in to_trans]
        
        time.sleep(15) #To make sure we don't time out the API
        
        #Translate back to source language
        from_trans = translate_client.translate(to_trans_list, target_language='en',
                                                source_language=args.lang)
        from_trans_list = [translation['translatedText'] for translation in from_trans]
        
        #Put it into our initialized vector
        textsToReturn[(divide_length*i):(divide_length*(i+1))] = from_trans_list
        
        time.sleep(15) #To make sure we don't time out the API

    print("Translation complete")
    
    #Create new Dataframe and save it
    newTargets = np.concatenate((targets, targets))
    newTexts = np.concatenate((texts, textsToReturn))
    
    newData = pd.DataFrame({'target': newTargets, 'text': newTexts})
    newData.to_csv("backtranslation_"+args.lang+".csv", index=False)