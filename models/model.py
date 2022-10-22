from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch.autograd import *
from .modules import masked_mean, masked_max, Attention, CoAttention, MaskedSoftmax, MyMultiHeadAttention
from transformers import AutoTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoModel, AutoTokenizer

class InformationFusion(nn.Module):
    def __init__(self, hidden_size):
        super(InformationFusion, self).__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
    def forward(self, query_hidden_state, tgt_hidden_state):
        merge_representation = torch.cat((query_hidden_state, tgt_hidden_state), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_hidden = torch.mul(gate_value, tgt_hidden_state)
        return gated_converted_hidden


def get_multi_head_att_paras():
    # default setting
    # n_head = 4
    # d_kv = 256
    # stack_num = 1
    # for seg in cur_model.split('_'):
    #     if seg[0] == 'h' and seg[1:].isdigit():
    #         n_head = int(seg[1:])
    #     if seg[0] == 'd' and seg[1:].isdigit():
    #         d_kv = int(seg[1:])
    #     if seg[0] == 'x' and seg[1:].isdigit():
    #         stack_num = int(seg[1:])
    #
    # print('\nStacked %d multi-head attention layer with head num: %d, dim: %d' % (stack_num, n_head, d_kv))
    n_head = 6
    d_kv = 128
    stack_num = 4
    return n_head, d_kv, stack_num

class MultimodalEncoder(nn.Module):
    def __init__(self, opt):
        """Initialize model."""
        super(MultimodalEncoder, self).__init__()
        self.opt = opt
        self.second_mode = opt.second_mode
        self.device = opt.device
        self.bert_hidden_size = opt.bert_hidden_size
        self.fc_feat_size = opt.fc_feat_size

        self.bert_layer_num = opt.bert_layer_num
        self.dropout = nn.Dropout(opt.dropout)
        self.init_bert(self.bert_layer_num)
        self.dropout = nn.Dropout(p=opt.dropout)
        n_head, d_kv, stack_num = get_multi_head_att_paras()

        if opt.first_mode in ['img_text', 'caption_text']:
            if opt.second_mode in ['attention', 'multihead', 'co_att']:
                self.linear_classifer_final = nn.Linear(opt.bert_hidden_size, opt.trg_class)
            if opt.second_mode in ['concat', 'multihead_text', 'attention_text']:
                self.linear_classifer_final = nn.Linear(opt.bert_hidden_size*2, opt.trg_class)

            self.two_modality_attention = Attention(opt.bert_hidden_size)
            self.two_modality_multi_attention = nn.ModuleList(
            [MyMultiHeadAttention(n_head, self.bert_hidden_size, d_kv, dropout=opt.dropout, need_mask=False)
             for _ in range(stack_num)])
            self.two_modality_co_attention = CoAttention(opt.bert_hidden_size, opt.bert_hidden_size)



        if opt.first_mode in ['img_text_caption']:
            if opt.second_mode in ['attention', 'multihead', 'co_att']:
                self.linear_classifer_final = nn.Linear(opt.bert_hidden_size*2, opt.trg_class)
            if opt.second_mode in ['concat', 'multihead_text', 'attention_text']:
                self.linear_classifer_final = nn.Linear(opt.bert_hidden_size*3, opt.trg_class)
            self.text2img_attention = Attention(opt.bert_hidden_size)
            self.text2cap_attention = Attention(opt.bert_hidden_size)
            self.text2cap_multi_attention = nn.ModuleList(
            [MyMultiHeadAttention(n_head, self.bert_hidden_size, d_kv, dropout=opt.dropout, need_mask=True)
             for _ in range(stack_num)])
            self.text2img_multi_attention = nn.ModuleList(
                [MyMultiHeadAttention(n_head, self.bert_hidden_size, d_kv, dropout=opt.dropout, need_mask=False)
                 for _ in range(stack_num)])
            self.text_img_co_attention = CoAttention(opt.bert_hidden_size, opt.bert_hidden_size)
            self.text_cap_co_attention = CoAttention(opt.bert_hidden_size, opt.bert_hidden_size)



        self.project_img_att_feat = nn.Linear(self.fc_feat_size, opt.bert_hidden_size)
        self.project_img_fc_feat = nn.Linear(self.fc_feat_size, opt.bert_hidden_size)
        self.linear_text_att = nn.Linear(self.bert_hidden_size, opt.bert_hidden_size)
        self.linear_cap_att = nn.Linear(self.bert_hidden_size, opt.bert_hidden_size)
        self.linear_text_fc = nn.Linear(self.bert_hidden_size, opt.bert_hidden_size)
        self.linear_text2cap_fc = nn.Linear(self.bert_hidden_size, opt.bert_hidden_size)
        self.linear_text2img_fc = nn.Linear(self.bert_hidden_size, opt.bert_hidden_size)

        self.linear_cap_fc = nn.Linear(self.bert_hidden_size, opt.bert_hidden_size)
        self.linear_multihead_text_fc = nn.Linear(self.bert_hidden_size, opt.bert_hidden_size)


    def init_bert(self, layer_num):
        bert_version = "vinai/bertweet-base"
        self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
        # since the framework of bertweet is same with  RoBERTa, so we directly use the RobertaConfig
        # reference: https://github.com/VinAIResearch/BERTweet/issues/17
        config = RobertaConfig.from_pretrained(bert_version)
        config.num_hidden_layers = layer_num
        self.bert_encoder = AutoModel.from_pretrained(bert_version, config=config).to(self.device)  # auto skip unused layers
    # def init_bert(self, layer_num):
    #     # bert_version = "vinai/bertweet-base"
    #     bert_version = "bert-base-uncased"
    #
    #     self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
    #     # since the framework of bertweet is same with  RoBERTa, so we directly use the RobertaConfig
    #     # reference: https://github.com/VinAIResearch/BERTweet/issues/17
    #     # config = RobertaConfig.from_pretrained(bert_version)
    #     config = BertConfig.from_pretrained(bert_version)
    #     config.num_hidden_layers = layer_num
    #     self.bert_encoder = AutoModel.from_pretrained(bert_version, config=config).to(
    #         self.device)  # auto skip unused layers




    def get_text_feat(self, memory_bank, text_pooling_type='max', mask=None):
        # map memory bank into one feat vector using mask
        assert len(memory_bank.shape) == 3

        if text_pooling_type == 'max':
            text_feats = masked_max(memory_bank, mask, dim=1)
        elif text_pooling_type == 'avg':
            text_feats = masked_mean(memory_bank, mask, dim=1)
        return text_feats

    def encode_text(self, texts):
        # texts_new = [text for text in texts]

        texts_new = texts
        input = self.tokenizer(texts_new, padding=True, truncation=True, max_length=50,
                               return_tensors="pt").to(self.device)
        attention_mask = list(input.values())[-1]
        out_states = self.bert_encoder(**input)[0]  # [0] is the last_hidden_state, [1] is the pooled_output
        return out_states, attention_mask

    def encode_img(self, att_feats):
        # read image visual feature and map them to bi_hidden_size
        # img_feats: [batch, 2048] for resnet152
        batch_size = att_feats.size(0)
        feat_size = att_feats.size(-1)
        att_feats = att_feats.view(batch_size, -1, feat_size)
        fc_feats = torch.mean(att_feats, dim=1)
        att_feats = self.project_img_att_feat(att_feats)
        fc_feats = self.project_img_fc_feat(fc_feats)

        return fc_feats, att_feats




    def forward(self, att_feats, texts, captions):
        fc_img_feats, att_img_feats = self.encode_img(att_feats)
        text_states, text_mask = self.encode_text(texts)
        caption_states, caption_mask = self.encode_text(captions)

        fc_text_ = self.get_text_feat(text_states, text_pooling_type='max', mask=text_mask)
        fc_text = self.linear_text_fc(fc_text_)
        fc_text2cap = self.linear_text2cap_fc(fc_text_)
        fc_text2img = self.linear_text2img_fc(fc_text_)


        fc_cap = self.get_text_feat(caption_states, text_pooling_type='max', mask=caption_mask)
        fc_cap = self.linear_cap_fc(fc_cap)
        att_cap = self.linear_cap_att(caption_states)

        if self.opt.first_mode == 'img_text':
            if self.second_mode == 'concat':
                combined_feat = torch.cat((fc_text, fc_img_feats), dim=1)
                predict = self.linear_classifer_final(combined_feat)
            if self.second_mode == 'attention':
                att_text2img = self.two_modality_attention(fc_text, att_img_feats)
                predict = self.linear_classifer_final(att_text2img)
            if self.second_mode == 'attention_text':
                att_text2img = self.two_modality_attention(fc_text2img, att_img_feats)
                predict = self.linear_classifer_final(torch.cat((att_text2img, fc_text), dim=1))
            if self.second_mode == 'co_att':
                att_co_feat = self.two_modality_co_attention(text_states, att_img_feats, text_mask )
                predict = self.linear_classifer_final(att_co_feat)
            if self.second_mode == 'multihead':
                enc_text2img = fc_text
                for enc_layer in self.two_modality_multi_attention:
                    enc_text2img, _ = enc_layer(enc_text2img, att_img_feats, att_img_feats)
                predict = self.linear_classifer_final(enc_text2img)
            if self.second_mode == 'multihead_text':
                enc_text2img = fc_text2img
                for enc_layer in self.two_modality_multi_attention:
                    enc_text2img, _ = enc_layer(enc_text2img, att_img_feats, att_img_feats)
                predict = self.linear_classifer_final(torch.cat((enc_text2img, fc_text), dim=1))


        if self.opt.first_mode == 'caption_text':
            if self.second_mode == 'concat':
                combined_feat = torch.cat((fc_text, fc_cap), dim=1)
                predict = self.linear_classifer_final(combined_feat)
            if self.second_mode == 'attention':
                att_text2cap = self.two_modality_attention(fc_text, att_cap, mask=caption_mask)
                predict = self.linear_classifer_final(att_text2cap)
            if self.second_mode == 'attention_text':
                att_text2cap = self.two_modality_attention(fc_text2cap, att_cap, mask=caption_mask)
                predict = self.linear_classifer_final(torch.cat((att_text2cap, fc_text), dim=1))
            if self.second_mode == 'co_att':
                att_co_feat = self.two_modality_co_attention(text_states, caption_states, text_mask, caption_mask)
                predict = self.linear_classifer_final(att_co_feat)
            if self.second_mode == 'multihead':
                # fc_text_ = self.dropout(fc_text)
                enc_text2cap = fc_text
                for enc_layer in self.two_modality_multi_attention:
                    enc_text2cap, _ = enc_layer(enc_text2cap, att_cap,att_cap, mask=caption_mask)
                predict = self.linear_classifer_final(enc_text2cap)
            if self.second_mode == 'multihead_text':
                # fc_text_ = self.dropout(fc_text)
                enc_text2cap = fc_text2cap
                for enc_layer in self.two_modality_multi_attention:
                    enc_text2cap, _ = enc_layer(enc_text2cap, att_cap, att_cap, mask=caption_mask)
                predict = self.linear_classifer_final(torch.cat((enc_text2cap, fc_text), dim=1))




        if self.opt.first_mode == 'img_text_caption':
            if self.second_mode == 'concat':
                combined_feat = torch.cat((fc_text, fc_cap, fc_img_feats), dim=1)
                predict = self.linear_classifer_final(combined_feat)
            if self.second_mode == 'attention':
                att_text2cap = self.text2cap_attention(fc_text, att_cap, mask=caption_mask)
                att_text2img = self.text2img_attention(fc_text, att_img_feats)

                predict = self.linear_classifer_final(torch.cat((att_text2cap, att_text2img), dim=1))
            if self.second_mode == 'attention_text':
                att_text2cap = self.text2cap_attention(fc_text2cap, att_cap, mask=caption_mask)
                att_text2img = self.text2img_attention(fc_text2img, att_img_feats)
                predict = self.linear_classifer_final(torch.cat((att_text2cap, att_text2img, fc_text), dim=1))
            if self.second_mode == 'co_att':
                img_text_att_co_feat = self.text_img_co_attention(text_states, att_img_feats, text_mask)
                cap_text_att_co_feat = self.text_cap_co_attention(text_states, caption_states, text_mask,
                                                                      caption_mask)
                predict = self.linear_classifer_final(torch.cat((img_text_att_co_feat, cap_text_att_co_feat), dim=1))


            if self.second_mode == 'multihead':
                enc_text2cap = fc_text2cap
                for enc_layer in self.text2cap_multi_attention:
                    enc_text2cap, _ = enc_layer(enc_text2cap, att_cap, att_cap, mask=caption_mask)

                enc_text2img = fc_text2img
                for enc_layer in self.text2img_multi_attention:
                    enc_text2img, _ = enc_layer(enc_text2img, att_img_feats, att_img_feats)

                predict = self.linear_classifer_final(torch.cat((enc_text2cap, enc_text2img), dim=1))
            if self.second_mode == 'multihead_text':
                enc_text2cap = fc_text2cap
                for enc_layer in self.text2cap_multi_attention:
                    enc_text2cap, _ = enc_layer(enc_text2cap, att_cap, att_cap, mask=caption_mask)

                enc_text2img = fc_text2img
                for enc_layer in self.text2img_multi_attention:
                    enc_text2img, _ = enc_layer(enc_text2img, att_img_feats, att_img_feats)

                predict = self.linear_classifer_final(torch.cat((enc_text2cap, enc_text2img, fc_text), dim=1))
        return predict