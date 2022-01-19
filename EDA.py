import pandas as pd

train_file = pd.read_csv('./data/train.csv')
job_tag = pd.read_csv('./data/job_tags.csv')

a = train_file.copy()
b = pd.merge(a, job_tag, on='jobID', how='outer')

user = set(train_file['userID'].tolist())
job = set(train_file['jobID'].tolist())

train_dummy = pd.get_dummies(train_file)

