import os
import numpy as np
import opts
import torch
import utils
import logging
from models.model import MultimodalEncoder
from dataloader import load_ce_data, load_all_data
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

def eval(opt):
    train_data, val_data, test_data = load_all_data(opt.text_file_path)
    test_dataset = load_ce_data(test_data, opt.img_feature_path)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=opt.train_batch_size, num_workers=4)

    model = MultimodalEncoder(opt).to(opt.device)
    model.eval()
    model_save_path = os.path.join(opt.save_path, 'model_best.pth')
    model.load_state_dict(torch.load(model_save_path))

    predictions, true_labels = [], []

    for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
        img_features, texts, labels, captions = batch


        img_features = img_features['feat']

        img_features = img_features.to(opt.device)
        labels = labels.to(opt.device)
        with torch.no_grad():
            predict_out = model(img_features, texts, captions)

        predictions.append(predict_out.detach().cpu().numpy())
        true_labels.extend(labels.to('cpu').numpy())

    predict_all = np.vstack(predictions)
    label_all = np.stack(true_labels)
    f_score, w_f1 = utils.calculate_score_f1(predict_all, label_all)
    test_f1 = w_f1
    print()
    print("label_0 F1  | label_1 F1  | label_2 F1  | label_3 F1  | label_4 F1  | Test f1  ")
    print(
        f"|   {f_score[0]:.4f}    |   {f_score[1]:.4f}   |   {f_score[2]:.4f}   |  {f_score[3]:.4f}    |  {f_score[4]:.4f}    |    {test_f1:.4f}")



if __name__ == "__main__":
    opt = opts.parse_opt()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt.text_file_path = './data/social_text_all_new.json'
    opt.img_feature_path = '/home/sdb_pro/EMNLP_discourse_code/final_dataset_features_att'
    eval(opt)