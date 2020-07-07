import pandas as pd
from torch.utils import data
import os 
import os.path as osp
import torch 
import sys


class ValidationQuestionReader(data.Dataset):

    def __init__(self, root,):
        self.root = root
        self.pth = osp.join(self.root, "validation.csv")
        self.pair_questions = pd.read_csv(self.pth)

    def __len__(self):
        return len(self.pair_questions)

    def __getitem__(self, index):
        batch_pair_questions = self.pair_questions.iloc[index,1:]
        question1 = batch_pair_questions['question1']
        question2 = batch_pair_questions['question2']
        is_duplicate = batch_pair_questions['is_duplicate']
        return question1,question2,is_duplicate

class QuestionReader(data.Dataset):

    def __init__(self, root,):
        self.root = root
        self.pth = osp.join(self.root, "train.csv")
        self.pair_questions = pd.read_csv(self.pth)

    def __len__(self):
        return len(self.pair_questions)

    def __getitem__(self, index):
        batch_pair_questions = self.pair_questions.iloc[index,1:]
        question1 = batch_pair_questions['question1']
        question2 = batch_pair_questions['question2']
        is_duplicate = batch_pair_questions['is_duplicate']
        return question1,question2,is_duplicate

#dst = QuestionReader(root = "/sdd1/amit/DAV/project/dataset")
# trainloader = data.DataLoader(dst, batch_size=2)

# if __name__ == '__main__':
#     for i, data in enumerate(trainloader):
#         question1, question2, labels = data
#         if i == 0:
#             # print(data)
#             # print(question1)
#             sys.exit()
