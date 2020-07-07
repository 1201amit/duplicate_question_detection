####################reference#########################
# https://mccormickml.com/2019/07/22/BERT-fine-tuning/
######################################################

from transformers import BertTokenizer
import torch 
import dataloader 
from torch.utils import data
import sys
from transformers import BertModel
import torch.optim as optim
import torch.nn as nn 
import os 
import numpy as np 
from itertools import cycle
import time 
from transformers import BertForSequenceClassification

device = torch.device('cpu') # set the device to cpu
if(torch.cuda.is_available()): # check if cuda is available
    device = torch.device('cuda:0') # if cuda, set device to cuda

def computeAccuracy(y_pred, y_gt):
    y_pred = y_pred.cpu().detach().numpy()
    y_gt = y_gt.cpu().numpy()
    y_pred = np.where(y_pred>0, 1.0, 0.0)
    len_correct_predict = len(np.argwhere(y_pred == y_gt))
    accuracy = len_correct_predict/len(y_pred)
    return accuracy 

def computeValidationAccuracy(bert_model, tokenizer, max_length=300):
    bert_model.eval()
    accuracy_vector = [] 

    dst = dataloader.ValidationQuestionReader(root="/sdd1/amit/DAV/project/dataset/split",)
    batch_size = 8

    validation_loader = data.DataLoader(dst, batch_size=batch_size, shuffle=True, drop_last=True,)
    print("validation started ") 
    for i, batch in enumerate(validation_loader):
        question1, question2, all_labels = batch
        all_labels = all_labels.numpy()
        labels = []
        input_ids = []
        attention_masks = []
        
        for b in range(batch_size):
            try: 
                encoded_dict = tokenizer.encode_plus(
                                question1[b], question2[b],         # Sentence to encode.
                                add_special_tokens = True,          # Add '[CLS]' and '[SEP]'
                                max_length = max_length,            # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,       # Construct attn. masks.
                                return_tensors = 'pt',              # Return pytorch tensors.
                                )

                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
                labels.append(all_labels[b])
                
            except ValueError: 
                pass

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attention_masks = torch.cat(attention_masks, dim=0).to(device)
        labels = torch.from_numpy(np.asarray(labels))
        labels = labels.view(-1,1).to(device=device, dtype=torch.long)
        
        loss, logits = bert_model(input_ids,
                        token_type_ids=None,
                        attention_mask=attention_masks,
                        labels=labels)

        y_pred = torch.argmax(logits, axis=1)
        labels = labels.view(-1)
        accuracy = computeAccuracy(y_pred, labels)
        accuracy_vector.append(accuracy)

    accuracy_vector = np.asarray(accuracy_vector)
    accuracy = np.mean(accuracy_vector)

    return accuracy 

def main():
    bert_model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",            # Use the 12-layer BERT model, with an uncased vocab.
                num_labels = 2,                 # The number of output labels--2 for binary classification.
                output_attentions = False,      # Whether the model returns attentions weights.
                output_hidden_states = False,   # Whether the model returns all hidden-states.
                )
    bert_model.to(device)
    
    bert_model.train()
    bert_model.load_state_dict(torch.load(os.path.join('checkpoints', 'bert_model')))

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    max_length = 512 # maximum tokenized length for each sentence. Maximum_token_length was 286. So I used 300. This part of code is deleted.
    # to directly get the token of sentence, use this command: <tokenizer.encode(sentence, add_spacial_tokens=True)>
    
    accuracy = computeValidationAccuracy(bert_model, tokenizer, max_length=512)
    print("validation accuracy is ", accuracy)
    


if __name__ == '__main__':
    main()
