import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--text_file_path', type=str, default='./data/social_text_all_new.json',
                    help='path to the json file containing the dataset')
    parser.add_argument('--img_feature_path', type=str, default='/home/xcp/code/EMNLP_disourse_code/final_dataset_features_att',
                    help='path to the directory containing the preprocessed att feats')

    parser.add_argument('--save_path', type=str,
                        default='./save',
                        help='path to save the model')
    # Model settings
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                    help='the hidden size of BERTweet')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                        help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--bert_layer_num', type=int, default=6,
                    help='number of layers in the BERT')
    parser.add_argument('--trg_class', type=int, default=5,
                        help='num of target class')


    # Optimization: General
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='the ratio to warm up the training of the BERT')
    parser.add_argument('--seed', type=int, default=26,
                        help='random seed')
    parser.add_argument('--train_batch_size', type=int, default=100,
                    help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='learning rate')


    parser.add_argument('--max_epochs', type=int, default=10,
                    help='number of epochs')
    parser.add_argument('--first_mode', type=str, default='img_text_caption',
                        help='choose the first mode from [img_text, caption_text, img_text_caption]')
    parser.add_argument('--second_mode', type=str, default='multihead_text',
                        help='choose the second mode from [concat, attention, attention_text, multihead, multihead_text, co_att')

    args = parser.parse_args()



    return args