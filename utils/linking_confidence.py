import pandas as pd

train = pd.read_csv("../data/train_conf.csv")
valid = pd.read_csv("../data/valid_conf.csv")
test = pd.read_csv("../data/test_conf.csv")

train_conf100 = train[train['conf'] == 1][['target', 'text']]
valid_conf100 = valid[valid['conf'] == 1][['target', 'text']]
test_conf100 = test[test['conf'] == 1][['target', 'text']]

train_conf80 = train[train['conf'] >= .8][['target', 'text']]
valid_conf80 = valid[valid['conf'] >= .8][['target', 'text']]
test_conf80 = test[test['conf'] >= .8][['target', 'text']]

train_conf100.to_csv("../data/train_conf100.csv")
valid_conf100.to_csv("../data/valid_conf100.csv")
test_conf100.to_csv("../data/test_conf100.csv")
train_conf80.to_csv("../data/train_conf80.csv")
valid_conf80.to_csv("../data/valid_conf80.csv")
test_conf80.to_csv("../data/test_conf80.csv")
