import pandas as pd
import numpy as np
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print(train)
#print(test)
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
#print (train.shape, test.shape, data.shape)

data.apply(lambda x: sum(x.isnull()))
data.describe
print(data.apply(lambda x: len(x.unique())))

