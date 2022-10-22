from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
from models.model import MultimodalEncoder
from dataloader import load_ce_data, load_all_data
import utils
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from optimization import BertAdam, warmup_linear
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import functools
import operator



def train(opt, logging):
    logging.info("***********************************************************************************************")
    logging.info("run experiments with seed {}, {}".format(opt.seed, opt.second_mode))
    train_data, val_data, test_data = load_all_data(opt.text_file_path)
    train_dataset = load_ce_data(train_data, opt.img_feature_path)
    dev_dataset = load_ce_data(val_data, opt.img_feature_path)
    test_dataset = load_ce_data(test_data, opt.img_feature_path)

    model = MultimodalEncoder(opt).to(opt.device)

    # Assure in training mode
    model.train()

    crit = utils.ClassCriterion()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(
        len(train_dataset) / opt.train_batch_size) * opt.max_epochs

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.learning_rate,
                         warmup=opt.warmup_proportion,
                         t_total=num_train_optimization_steps)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.train_batch_size, num_workers=4)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=opt.train_batch_size, num_workers=4)


    logging.info("***** Running training *****")
    best_val_score = 0
    for train_idx in trange(int(opt.max_epochs), desc="Epoch"):
        logging.info("********** Epoch: " + str(train_idx) + " **********")
        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Batch size = %d", opt.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        model.train()
        train_total_loss, train_total_len, train_num_correct = 0, 0, 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            img_features, texts, labels, captions = batch

            img_features = img_features['feat']
            train_total_len += img_features.shape[0]
            img_features = img_features.to(opt.device)
            labels = labels.to(opt.device)
            predict_out = model(img_features, texts, captions)
            loss = crit(predict_out, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_total_loss += loss.item()
            pred = predict_out.argmax(1, keepdim=True).float()
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            train_num_correct += np.sum(correct)

        train_acc = train_num_correct / train_total_len
        avg_train_loss = train_total_loss / train_total_len

        logging.info("###############################")
        logging.info("Running Validation...")
        logging.info("###############################")
        print("###############################")
        print("Running Validation...")
        print("###############################")
        val_total_loss, val_total_len, val_num_correct = 0, 0, 0

        model.eval()
        predictions, true_labels = [], []

        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            img_features, texts, labels, captions = batch

            img_features = img_features['feat']

            val_total_len += img_features.shape[0]
            img_features = img_features.to(opt.device)
            labels = labels.to(opt.device)
            with torch.no_grad():
                predict_out = model(img_features, texts, captions)
            loss = crit(predict_out, labels)

            val_total_loss += loss.item()
            pred = predict_out.argmax(1, keepdim=True).float()
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            val_num_correct += np.sum(correct)

            predictions.append(predict_out.detach().cpu().numpy())
            true_labels.extend(labels.to('cpu').numpy())
        predict_all = np.vstack(predictions)
        label_all = np.stack(true_labels)
        f_score, w_f1 = utils.calculate_score_f1(predict_all, label_all)
        val_f1 = w_f1

        val_acc = val_num_correct / val_total_len
        avg_val_loss = val_total_loss / val_total_len
        print()
        logging.info("Epoch | Train Accuracy  | label_0 F1  | label_1 F1  | label_2 F1  | label_3 F1  | label_4 F1  | Validation F1  | Training Loss | Validation Loss")
        logging.info(
            f"{train_idx + 1:4d} |  {train_acc:.4f}  |    {f_score[0]:.4f}     |    {f_score[1]:.4f}      |    {f_score[2]:.4f}     |    {f_score[3]:.4f}    |    {f_score[4]:.4f}    |   {val_f1:.4f}      |    {avg_train_loss:.4f}    |     {avg_val_loss:.4f}")
        print(
            "Epoch | Train Accuracy  | label_0 F1  | label_1 F1  | label_2 F1  | label_3 F1  | label_4 F1  | Validation F1  | Training Loss | Validation Loss")
        print(
            f"{train_idx + 1:4d} |  {train_acc:.4f}  |    {f_score[0]:.4f}     |    {f_score[1]:.4f}      |    {f_score[2]:.4f}     |    {f_score[3]:.4f}    |    {f_score[4]:.4f}    |   {val_f1:.4f}      |    {avg_train_loss:.4f}    |     {avg_val_loss:.4f}")


        model_save_path = os.path.join(opt.save_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # save the best model
        if val_f1 > best_val_score:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'model_best.pth'))
            best_val_score = val_f1
            logging.info("**********best val************")



