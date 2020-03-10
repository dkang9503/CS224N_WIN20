from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import glob
import re

def main():
    df = pd.DataFrame()
    
    modelList = ['BERT', 'elmo', 'baseline'] # For each model directory in logs
    counter = 0
    for model in modelList:    
        listOfRuns = glob.glob(f'../logs/{model}/*') # get all the runs
        for run in listOfRuns: #For each run
            print(counter)
            files = glob.glob(run + '/*')
            info_dict, file_name = returnParamsDict(files[0], model)
            info_dict['file'] = file_name
            for file in files:
                summaryIterator = summary_iterator(file)        
                for summary in summaryIterator:
                    if summary.summary.value._values != []:                
                        info = summary.summary.value._values[0]
                        if info.tag in info_dict.keys():                
                            info_dict[info.tag] += [info.simple_value]
                        else:
                            info_dict[info.tag] = [info.simple_value]
            df = df.append([info_dict])        
            counter+=1    

def returnParamsDict(file, model):
    parsed_file = re.search(f'{model}\\\\(.*)_lr', file).group(1)
    params = {}
    param_list = ['backtranslation_ja', 'backtranslation_es', 'backtranslation_de',
                  'ri', 'rs', 'sr', 'rd'] #add more later 
    for param in param_list:
        if param in parsed_file:
            params[param] = 1
        else:
            params[param] = 0
    
    return params, parsed_file

if __name__ == "__main__()":
    main()