from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


import six
from six.moves import cPickle
from torch.autograd import Variable
import logging
import os
from sklearn.metrics import precision_recall_fscore_support as score
import random
import torch.nn.functional as F

from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None:
                continue
            else:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# class ClassCriterion(nn.Module):
#     def __init__(self):
#         super(ClassCriterion, self).__init__()
#         self.criterion = nn.CrossEntropyLoss()
#     def forward(self, classifier_output, trg_class, class_num):
#         classifier_loss = self.criterion(classifier_output, trg_class.squeeze())
#
#         return classifier_loss

class ClassCriterion(nn.Module):
    def __init__(self):
        super(ClassCriterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)
    def forward(self, classifier_output, trg_class):
        batch = trg_class.size(0)
        label_0 = batch / (sum(torch.eq(trg_class.squeeze(), 0).int()).item() + 1)
        label_1 = batch / (sum(torch.eq(trg_class.squeeze(), 1).int()).item() + 1)
        label_2 = batch / (sum(torch.eq(trg_class.squeeze(), 2).int()).item() + 1)
        label_3 = batch / (sum(torch.eq(trg_class.squeeze(), 3).int()).item() + 1)
        label_4 = batch / (sum(torch.eq(trg_class.squeeze(), 4).int()).item() + 1)
        label_weigt_sum = label_0 + label_1 + label_2 + label_3 + label_4
        label_weight_0 = label_0 / label_weigt_sum
        label_weight_1 = label_1 / label_weigt_sum
        label_weight_2 = label_2 / label_weigt_sum
        label_weight_3 = label_3 / label_weigt_sum
        label_weight_4 = label_4 / label_weigt_sum
        weight = Variable(classifier_output.data.new(batch).zero_())
        label_0_idx = (trg_class.squeeze() == 0).nonzero().squeeze()
        label_1_idx = (trg_class.squeeze() == 1).nonzero().squeeze()
        label_2_idx = (trg_class.squeeze() == 2).nonzero().squeeze()
        label_3_idx = (trg_class.squeeze() == 3).nonzero().squeeze()
        label_4_idx = (trg_class.squeeze() == 4).nonzero().squeeze()
        weight[label_0_idx] = label_weight_0
        weight[label_1_idx] = label_weight_1
        weight[label_2_idx] = label_weight_2
        weight[label_3_idx] = label_weight_3
        weight[label_4_idx] = label_weight_4


        classifier_loss = self.criterion(classifier_output, trg_class.squeeze())
        classifier_loss = torch.mean(classifier_loss * weight)
        return classifier_loss





def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def calculate_score(predict, target):
    # predict: [batch, class_size], target:[batch,1]
    predict_label = predict.argmax(axis=1)
    target = target.squeeze()
    precision, recall, fscore, support = score(target, predict_label)
    return precision, recall, fscore, support


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def metrics(labels, preds, argmax_needed: bool = False):
    """
    Returns the Matthew's correlation coefficient, accuracy rate, true positive rate, true negative rate, false positive rate, false negative rate, precission, recall, and f1 score

    labels: list of correct labels

    pred: list of model predictions

    argmax_needed (boolean): converts logits to predictions. Defaulted to false.
    """

    if argmax_needed == True:
        preds = np.argmax(preds, axis=1).flatten()

    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    results = {
        "mcc": mcc,
        "acc": acc,
        # "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # return results, labels, preds
    return results




def reshape_text(texts, num):
    # texts: [num, batch] -> [batch, num] -> [batch*num]
    batch = len(texts[0])
    texts_new = []
    for i in range(batch):
        tmp = []
        for j in range(num):
            tmp.append(texts[j][i])
        texts_new.extend(tmp)
    return texts_new


def restrict_label(predict_t):
    # predict_t:[batch, tag_num]
    batch = predict_t.size(0)
    pre_label = torch.argmax(predict_t, dim=1)
    mask = torch.zeros(batch, device=pre_label.device)
    cnt = 0
    for ptr, l in enumerate(pre_label):
        l = l.item()
        if l == 0:
           cnt+=1
           if cnt < int(0.5*batch):
               mask[ptr] = 1
        else:
            mask[ptr] = 1
    return mask

def calculate_score_f1(predict_all, label_all):
    precision, recall, fscore, support = calculate_score(predict_all, label_all)
    label_all_ = np.squeeze(label_all)
    label_0 = sum(np.equal(label_all_, 0))
    label_1 = sum(np.equal(label_all_, 1))
    label_2 = sum(np.equal(label_all_, 2))
    label_3 = sum(np.equal(label_all_, 3))
    label_4 = sum(np.equal(label_all_, 4))
    label_sum = label_0 + label_1 + label_2 + label_3 + label_4
    score_weight = np.array(
        [label_0 / label_sum, label_1 / label_sum, label_2 / label_sum, label_3 / label_sum, label_4 / label_sum])
    overall_fscore = score_weight[0] * fscore[0] + score_weight[1] * fscore[1] + score_weight[2] * fscore[2] \
                     + score_weight[3] * fscore[3] + score_weight[4] * fscore[4]
    return fscore, overall_fscore


