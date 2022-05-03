import logging
import random, os
from bert_exp.joint_bert_ic_sf.args import arg
import torch
import numpy as np
from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
from bert_exp.joint_bert_ic_sf.model import JointBERT, JointDistilBERT, JointAlbert
from seqeval.metrics import precision_score, recall_score, f1_score


MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}

def load_tokenizer():
    return MODEL_CLASSES[arg["model_type"]][2].from_pretrained(MODEL_PATH_MAP[arg["model_type"]])


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed():
    random.seed(arg["seed"])
    np.random.seed(arg["seed"])
    torch.manual_seed(arg["seed"])
    if not arg["no_cuda"] and torch.cuda.is_available():
        torch.cuda.manual_seed_all(arg["seed"])

def get_intent_labels():
    return [label.strip() for label in open(os.path.join(arg["data_dir"], arg["task"], arg["intent_label_file"]), 'r', encoding='utf-8')]


def get_slot_labels():
    return [label.strip() for label in open(os.path.join(arg["data_dir"], arg["task"], arg["slot_label_file"]), 'r', encoding='utf-8')]

def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }

def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }

def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }

def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results
