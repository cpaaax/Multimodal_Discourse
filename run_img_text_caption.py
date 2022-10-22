import os

import opts
import torch
import utils
import logging
from train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def logging_set(mode):
    if not os.path.exists('./logging'):
        os.makedirs('./logging')
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    filemode = 'w',
                    filename='./logging/{}.log'.format(mode),
                    level = logging.INFO)
    return logging




if __name__ == "__main__":

    opt = opts.parse_opt()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt.text_file_path = './data/social_text_all_new.json'
    opt.img_feature_path = '/home/sdb_pro/EMNLP_discourse_code/final_dataset_features_att'

    utils.setup_seed(opt.seed)
    train(opt, logging=logging_set('train__{}_'.format(opt.first_mode, opt.second_mode)))

