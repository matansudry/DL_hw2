
"""
Here, we create a custom dataset
"""
import torch
import pickle
#import argparse
import os
#import sys
import json
#import numpy as np
#import re
import pickle
#import utils
import tqdm
from utils.types import PathT
import torch.utils.data as data
#from torch.utils.data import DataLoader
from typing import Any, Tuple, Dict, List
import torchvision.transforms as transforms
from PIL import Image
#from models.base_model import MyModel
#from torch.nn.utils.rnn import pack_padded_sequence
from utils.preprocess_vocab import preprocess_vocab_func
from utils.preprocess_answer import load_v2

class IamgesArchive(data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


# In[3]:


class MyDataset(data.Dataset):
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, image_path, questions_path, answers_path, train=True, answerable_only=False):#, answerable_only=False):
        # Set variables
        self.image_features_path = image_path
        self.questions_path = questions_path
        self.answers_path = answers_path
        
        #load the dataset of I, Q, A including the vocab of Q and A
        with open(questions_path, 'r') as fd:
            self.questions_json = json.load(fd)
            
        if (train):
            dataset_type = "train"
        else:
            dataset_type = "val"
            
        #preparing train and val quastion vocab
        if os.path.isfile("../data/cache/question_vocab_train") and os.path.isfile("../data/cache/question_vocab_val"):
            pass
        else:
            preprocess_vocab_func()
            
        #load question vocab
        with open("../data/cache/question_vocab_"+dataset_type, 'r') as fd:
            vocab_json = json.load(fd)
        

        
        #Vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab#['question']
        
        #preparing
        #preparing train and val quastion vocab
        if os.path.isfile("../data/cache/trainval_ans2label.pkl"):
            pass
        else:
            load_v2()
        
        
        with open("../data/cache/trainval_ans2label.pkl", "rb") as f:
            unpickler = pickle.Unpickler(f)
            # if file is not empty scores will be equal
            # to the value unpickled
            dict_answers = unpickler.load()
            self.number_of_answers_per_question = len(dict_answers)
        
        
        print("files upload was done")
        
        #load Q
        if os.path.isfile("../data/questions_"+dataset_type):
            self.questions = torch.load("../data/questions_"+dataset_type)
        else:
            self.questions = list(self.prepare_questions())
            self.questions = [self._encode_question(q, self.token_to_index) for q in self.questions] 
            torch.save(self.questions, "../data/questions_"+dataset_type)
        
        print("questions done")
        
        #change Q to Q dict    
        if os.path.isfile("../data/questions_dict_"+dataset_type):
            with open("../data/questions_dict_"+dataset_type, 'rb') as handle:
                self.questions_dict = pickle.load(handle)
        else:
            self.questions_dict = self.questions_to_dict()
            with open("../data/questions_dict_"+dataset_type, 'wb') as handle:
                pickle.dump(self.questions_dict, handle)
                
        print("questions dict done")
                
        #Load question_id_to_image_id
        if os.path.isfile("../data/question_id_to_image_id_"+dataset_type):
            with open("../data/question_id_to_image_id_"+dataset_type, 'r') as fd:
                self.question_id_to_image_id = json.load(fd)
        else:
            self.question_id_to_image_id = self.question_id_to_image_id()
            with open("../data/question_id_to_image_id_"+dataset_type, 'w') as fd:
                json.dump(self.question_id_to_image_id, fd)

        print("question_id_to_image_id done")
        
        
        #load A
        self.answerable_only = answerable_only
        if os.path.isfile("../data/answerable_with_labels_only_"+dataset_type+"_"+str(answerable_only)):
            with open("../data/answerable_with_labels_only_"+dataset_type+"_"+str(answerable_only), 'rb') as handle:
                self.answerable = pickle.load(handle)
        else:
            #preprocess A
            self.answerable = self.preprocess_answers(train)
            if self.answerable_only:
                self.answerable = self._find_answerable()
            with open("../data/answerable_with_labels_only_"+dataset_type+"_"+str(answerable_only), 'wb') as handle:
                pickle.dump(self.answerable, handle)
        
        print("answers done")
        
        #load I
        if os.path.isfile("../data/images_"+dataset_type):
            with open("../data/images_"+dataset_type, 'rb') as handle:
                self.images = pickle.load(handle)

        else:
            #preprocess A
            self.images = self.load_images()
            with open("../data/images_"+dataset_type, 'wb') as handle:
                pickle.dump(self.images, handle)
        
        print("images done")
        
        #load coco_images_to_dict
        if os.path.isfile("../data/coco_images_to_dict"+dataset_type):
            with open("../data/coco_images_to_dict"+dataset_type, 'rb') as handle:
                self.images_dict = pickle.load(handle)

        else:
            self.coco_images_to_dict()
            with open("../data/coco_images_to_dict"+dataset_type, 'wb') as handle:
                pickle.dump(self.images_dict, handle)
                
        print("coco_images_to_dict done")
                
        self.index_to_question_number_dict = self.index_to_question_number_func()
    
    def __getitem__(self, item):
        item = self.index_to_question_number_dict[item]
        q, q_length = self.questions_dict[item]
        a = self.answerable[item]
        temp = torch.zeros(self.number_of_answers_per_question)
        for answer_index in range(len(a[0])):
            temp[a[0][answer_index]] = a[1][answer_index]
        image_id = self.question_id_to_image_id[str(item)]
        image_id = self.images_dict[image_id]
        v = self.images[0][image_id][1]
        
        return v, temp, q, item, q_length

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.questions_dict)
    
    def get_transform(self, target_size, central_fraction=1.0):
        return transforms.Compose([
            transforms.Scale(int(target_size / central_fraction)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def load_images(self):
        transform = self.get_transform(target_size=224, central_fraction=0.875)
        dataset = [IamgesArchive(self.image_features_path, transform=transform)]
        return dataset
    
    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length
                    
    def preprocess_answers(self, train=True):
        if train:
            with open("../data/cache/train_target.pkl", "rb") as f:
                unpickler = pickle.Unpickler(f)
                scores = unpickler.load()
        else:
            with open("../data/cache/val_target.pkl", "rb") as f:
                unpickler = pickle.Unpickler(f)
                scores = unpickler.load()  

        with open("../data/cache/trainval_ans2label.pkl", "rb") as f:
            unpickler = pickle.Unpickler(f)
            dict_answers = unpickler.load()
            self.number_of_answers_per_question = len(dict_answers)

        answers_dict = {}
        for item in scores:
            answers_dict[item['question_id']] = ((item['labels'], item['scores']))

        return answers_dict
    
    def questions_to_dict(self):
        question_dict = {}
        for i in range(len(self.questions_json['questions'])):
            question_dict[self.questions_json['questions'][i]['question_id']] = self.questions[i] 
        return (question_dict)
    
    def question_id_to_image_id(self):
        question_id_dict = {}
        for i in range(len(self.questions_json['questions'])):
            question_id_dict[str(self.questions_json['questions'][i]['question_id'])] = self.questions_json['questions'][i]['image_id']
        return (question_id_dict)
    
    def _find_answerable(self):
        update_answers = self.answerable.copy()
        for answer in tqdm.tqdm(self.answerable):
            if (sum(self.answerable[answer])==0):
                del update_answers[answer]
        return(update_answers)
               
    def prepare_questions(self):
        """ Tokenize and normalize questions from a given question json in the usual VQA format. """
        questions = [q['question'] for q in self.questions_json['questions']]
        for question in questions:
            question = question.lower()[:-1]
            yield question.split(' ')

    def _encode_question(self, question, token_to_index):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def index_to_question_number_func(self):
        index_to_question_number_dict = {}
        cnt = 0
        for question in self.answerable:
            index_to_question_number_dict[cnt] = question
            cnt += 1
        return index_to_question_number_dict
    
    def coco_images_to_dict(self):
        images_dict= {}
        images = self.images[0]
        cnt = 0
        for image in tqdm.tqdm(images):
            images_dict[image[0]] = cnt
            cnt +=1
        self.images_dict = images_dict
    def num_tokens(self):
        return len(self.vocab) + 1





