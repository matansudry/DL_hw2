import json
from collections import Counter
import itertools
import config
import utils


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()[:-1]
        yield question.split(' ')

def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def preprocess_vocab_func():
    train_questions = "../../../datashare/v2_OpenEnded_mscoco_train2014_questions.json"
    val_questions = "../../../datashare/v2_OpenEnded_mscoco_val2014_questions.json"

    with open(train_questions, 'r') as file:
        train_questions = json.load(file)
    with open(val_questions, 'r') as file:
        val_questions = json.load(file)

    train_questions = prepare_questions(train_questions)
    train_question_vocab = extract_vocab(train_questions, start=1)
    
    val_questions = prepare_questions(val_questions)
    val_question_vocab = extract_vocab(val_questions, start=1)

    with open("../data/cache/question_vocab_train", 'w') as file:
        json.dump(train_question_vocab, file)
    with open("../data/cache/question_vocab_val", 'w') as file:
        json.dump(val_question_vocab, file)








