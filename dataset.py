"""
Here, we create a custom dataset
"""
import torch
import pickle
import argparse
import os
import sys
import json
import numpy as np
import re
import pickle
import utils
import tqdm
from utils.types import PathT
import torch.utils.data as data
from torch.utils.data import DataLoader
from typing import Any, Tuple, Dict, List
import torchvision.transforms as transforms
from PIL import Image
import h5py
from utils.image_preprocessing import image_preprocessing_master
from utils.preprocessing_vocab_questions import preprocessing_vocab

class MyDataset(data.Dataset):
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, image_path, questions_path, answers_path, train=True, answerable_only=False):
        # Set variables
        self.image_features_path = image_path
        self.questions_path = questions_path
        self.answers_path = answers_path
        self.padding_size = 22

        #load the dataset of I, Q, A including the vocab of Q and A
        with open(questions_path, 'r') as fd:
            self.questions_json = json.load(fd)

        if (train):
            dataset_type = "train"
        else:
            dataset_type = "val"

        self.dataset_type = dataset_type


        #load question vocab
        if os.path.isfile("../data/cache/trainval_q2label.pkl"):
            with open("../data/cache/trainval_q2label.pkl", "rb") as f:
                unpickler = pickle.Unpickler(f)
                vocab_json = unpickler.load()

        else:
            preprocessing_vocab()
            with open("../data/cache/trainval_q2label.pkl", "rb") as f:
                unpickler = pickle.Unpickler(f)
                vocab_json = unpickler.load()
            
        #target embedding
        if train == True:
            with open('../data/cache/train_target.pkl', "rb") as f:
                self.target = pickle.load(f)
        else:
            with open('../data/cache/val_target.pkl', "rb") as f:
                self.target = pickle.load(f)


        #Vocab
        self.vocab = vocab_json
        self.token_to_index = self.vocab

        with open("../data/cache/trainval_ans2label.pkl", "rb") as f:
            unpickler = pickle.Unpickler(f)
            dict_answers = unpickler.load()
            self.number_of_answers_per_question = len(dict_answers)


        print("files upload was done")

        #load Q
        if os.path.isfile("../data/questions_"+dataset_type):
            print("opening existing Qs")
            self.questions = torch.load("../data/questions_"+dataset_type)
        else:
            self.questions = list(self.prepare_questions())
            self.questions = [self._encode_question(q, self.token_to_index) for q in self.questions] 
            torch.save(self.questions, "../data/questions_"+dataset_type)

        print("questions done")

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
            with open("../data/answerable_with_labels_only_"+dataset_type+"_"+str(answerable_only), 'wb') as handle:
                pickle.dump(self.answerable, handle)

        print("answers done")

        if os.path.isfile("../data/cache/"+dataset_type+".h5"):
            with open("../data/cache/img2idx_"+dataset_type+".pkl", 'rb') as handle:
                self.img2idx = pickle.load(handle)

        else:
            image_preprocessing_master()
            with open("../data/cache/img2idx_"+dataset_type+".pkl", 'rb') as handle:
                self.img2idx = pickle.load(handle)

        print("images done")

    def __getitem__(self, item):
        target = self.target[item]
        q_id = target['question_id']
        answer_labels = target['labels']
        answer_scores = target['scores']
        ans = torch.zeros(self.number_of_answers_per_question)
        for answer_index in range(len(answer_labels)):
            ans[answer_labels[answer_index]] = answer_scores[answer_index]
        image_id = self.question_id_to_image_id[str(q_id)]
        images = h5py.File("../data/cache/"+self.dataset_type+".h5", 'r')
        image_index = self.img2idx[image_id]
        v = images['images'][image_index].astype('float32')
        v = torch.from_numpy(v)
        question = target['question_idx']
        question = self.__apply_padding(question)
        question = torch.tensor(question, dtype=torch.long)
        return (v, ans, question )

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.target)

    def get_transform(self, target_size, central_fraction=1.0):
        return transforms.Compose([
            transforms.Scale(int(target_size / central_fraction)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

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
            self.target = [t for t in self.target if t['label_counts']]
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

    def question_id_to_image_id(self):
        question_id_dict = {}
        for i in range(len(self.questions_json['questions'])):
            question_id_dict[str(self.questions_json['questions'][i]['question_id'])] = self.questions_json['questions'][i]['image_id']
        return (question_id_dict)

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

    def __apply_padding(self, question):
        if len(question) < self.padding_size:
            padding = [0] * (self.padding_size - len(question))
            question.extend(padding)
        return question