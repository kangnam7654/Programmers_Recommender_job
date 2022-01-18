import pandas as pd

train_file = pd.read_csv('./data/train.csv')

user = set(train_file['userID'].tolist())
job = set(train_file['jobID'].tolist())

train_dummy = pd.get_dummies(train_file)

