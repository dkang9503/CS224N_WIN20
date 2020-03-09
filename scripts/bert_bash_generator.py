import os
output_dir = "./"
output = "aug2_bert_train.sh"

files = os.listdir('../data/aug2_data/')
# remove extensions
files = [f[:-4] for f in files]

with open(output_dir + output, 'w') as f:
    f.write("#!/bin/bash\n")
    for fi in files:
        f.write(f"python ../train/bert_train.py --dataset=aug2_data/{fi}\n")

