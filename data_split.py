import pandas as pd 
import sys 
root = "/sdd1/amit/DAV/project/dataset/"
train_csv = "train.csv"

data = pd.read_csv(root+train_csv)
qid1 = pd.read_csv(root+train_csv, usecols= ['qid1'])
qid2 = pd.read_csv(root+train_csv, usecols= ['qid2'])
question1 = pd.read_csv(root+train_csv, usecols= ['question1'])
question2 = pd.read_csv(root+train_csv, usecols= ['question2'])
is_duplicate = pd.read_csv(root+train_csv, usecols= ['is_duplicate'])

#columns = ['qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
#training = pd.DataFrame([qid1, qid2, question1, question2, is_duplicate], columns=columns)
#training.to_csv(root+'training.csv')

validation_dict = {'qid1': qid1.iloc[0:10000,0], 'qid2': qid2.iloc[0:10000,0], 'question1': question1.iloc[0:10000,0], 'question2': question2.iloc[0:10000,0], 'is_duplicate': is_duplicate.iloc[0:10000,0]}  
train_dict = {'qid1': qid1.iloc[10000:,0], 'qid2': qid2.iloc[10000:,0], 'question1': question1.iloc[10000:,0], 'question2': question2.iloc[10000:,0], 'is_duplicate': is_duplicate.iloc[10000:,0]}  

validation = pd.DataFrame(validation_dict) 
validation.to_csv(root+'split/validation.csv')
train = pd.DataFrame(train_dict) 
train.to_csv(root+'split/train.csv')

#print(qid1.iloc[2, 0])

