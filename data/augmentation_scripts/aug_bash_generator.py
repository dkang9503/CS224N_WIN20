output_dir = "./"
output = "char_aug_bash_script.sh"

alpha_cs = [0.1, 0.3]
num_augs = [1, 3]
da_types = ['ocr', 'key']
t = "char"
input = '../train.csv'
output = '../aug2_data/'

with open(output_dir + t + '_aug_bash_script.sh', 'w') as f:
    f.write("#!/bin/bash\n")
    for da in da_types:
        for n in num_augs:
            for a_c in alpha_cs:
                s = f'python {t}_aug_script.py --input={input} --da_type={da} ' + \
                    f'--num_aug={n} --alpha_char={a_c} --output_dir={output}\n'
                f.write(s)

t = 'word'
alphas = [0.1, 0.3]
da_types = ['glovetwit', 'contextin', 'contextsub']
with open(output_dir + t + '_aug_bash_script.sh', 'w') as f:
    f.write("#!/bin/bash\n")
    for da in da_types:
        for n in num_augs:
            for a in alphas:
                s = f'python {t}_aug_script.py --input={input} --da_type={da} ' + \
                    f'--num_aug={n} --model_dir=../../models/saved_models/ ' + \
                    f'--output_dir={output} --alpha={a}\n'
                f.write(s)

t = 'sentence'
with open(output_dir + t + '_aug_bash_script.sh', 'w') as f:
    f.write("#!/bin/bash\n")
    for n in num_augs:
        s = f'python {t}_aug_script.py --input={input} --num_aug={n}' + \
                    f' --model_type=gpt2\n --output_dir={output}\n'
        f.write(s)
