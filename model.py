# -*- coding: utf-8 -*-#
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os

from transformers import RobertaForMaskedLM

from CGE import GAT

# Choose a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# -------------------------------------------------------------------------------
# Name:         model
# Description:
# Author:       梁超
# Date:         2024/5/14
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
answer_space_sub = [50273, 50274, 50275]
answer_space_temp = [50268, 50269, 50270]
answer_space_cau = [50265, 50266, 50267]
answer_space_cof = [50271, 50272]


class base(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.RoBERTa_MLM = RobertaForMaskedLM.from_pretrained(args.model_name)
        self.RoBERTa_MLM.resize_token_embeddings(args.vocab_size)
        for param in self.RoBERTa_MLM.parameters():
            param.requires_grad = True

        self.RoBERTa_MLM_for_schema = RobertaForMaskedLM.from_pretrained(args.model_name)
        self.RoBERTa_MLM_for_schema.resize_token_embeddings(args.vocab_size)
        for param in self.RoBERTa_MLM_for_schema.parameters():
            param.requires_grad = True

        self.hidden_size = 768
        self.active = nn.Tanh()
        self.to_sub = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_cau = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_temp = nn.Linear(self.hidden_size, self.hidden_size)
        self.to_cof = nn.Linear(self.hidden_size, self.hidden_size)

        self.GAT = GAT(self.hidden_size, self.hidden_size, args.GAT_drop, args.alpha, args.num_heads)

        self.mlp_sub = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 3))
        self.mlp_cau = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 3))
        self.mlp_temp = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 3))
        self.mlp_cof = nn.Sequential(nn.Linear(self.hidden_size, args.mlp_size),
                                 nn.ReLU(), nn.Dropout(args.mlp_drop),
                                 nn.Linear(args.mlp_size, 2))
        self.vocab_size = args.vocab_size
        self.mention_ratio = args.mention_weight
        self.type_ratio = args.type_weight

    def forward(self, idxs, masks, sent_ids, locs, mention_schema, mention_mask, mention_loc, type_schema, type_mask, type_loc, adjacency, question, device):
        # Mention schema feature construction
        mention_schema = self.RoBERTa_MLM_for_schema.roberta(mention_schema[0], attention_mask=mention_mask[0], output_hidden_states=True)[0]
        mention_mask = mention_schema[0][mention_loc]
        mention_pro = self.RoBERTa_MLM_for_schema.lm_head(mention_mask)
        mention_sub, mention_temp, mention_cau, mention_cof = mention_pro[:, answer_space_sub], mention_pro[:, answer_space_temp], mention_pro[:, answer_space_cau], mention_pro[:, answer_space_cof]

        # Type schema feature construction
        type_schema = self.RoBERTa_MLM_for_schema.roberta(type_schema[0], attention_mask=type_mask[0], output_hidden_states=True)[0]
        type_mask = type_schema[0][type_loc]
        type_pro = self.RoBERTa_MLM_for_schema.lm_head(type_mask)
        type_sub, type_temp, type_cau, type_cof = type_pro[:, answer_space_sub], type_pro[:, answer_space_temp], type_pro[:, answer_space_cau], type_pro[:, answer_space_cof]

        # Semantic feature construction
        all_sentences = self.RoBERTa_MLM.roberta(idxs[0], attention_mask=masks[0], output_hidden_states=True)[0].to(device)
        all_events = torch.zeros((len(locs), self.hidden_size)).to(device)

        for i in range(len(sent_ids)):
            all_events[i] = all_sentences[int(sent_ids[i])][int(locs[i])]

        adjacency = self.pro_adjacency(adjacency[0])

        all_events_sub = self.active(self.to_sub(all_events))
        all_events_cau = self.active(self.to_cau(all_events))
        all_events_temp = self.active(self.to_temp(all_events))
        all_events_cof = self.active(self.to_cof(all_events))

        # Attention
        ###################################################################################
        all_events_sub = self.GAT(all_events_sub, adjacency[0], adjacency[1]+adjacency[2]+adjacency[3]) + all_events_sub
        all_events_cau = self.GAT(all_events_cau, adjacency[2], adjacency[0]+adjacency[1]+adjacency[3]) + all_events_cau
        all_events_temp = self.GAT(all_events_temp, adjacency[1], adjacency[0]+adjacency[2]+adjacency[3]) + all_events_temp
        all_events_cof = self.GAT(all_events_cof, adjacency[3], adjacency[0]+adjacency[1]+adjacency[2]) + all_events_cof
        ###################################################################################

        sub_r = all_events_sub[question[0]] - all_events_sub[question[1]]
        cau_r = all_events_cau[question[0]] - all_events_cau[question[1]]
        temp_r = all_events_temp[question[0]] - all_events_temp[question[1]]
        cof_r = all_events_cof[question[0]] - all_events_cof[question[1]]

        # del all_events

        pro_sub = self.mlp_sub(sub_r)
        pro_cau = self.mlp_cau(cau_r)
        pro_temp = self.mlp_temp(temp_r)
        pro_cof = self.mlp_cof(cof_r)

        return self.fuse_probability(pro_sub, mention_sub, type_sub), self.fuse_probability(pro_cau, mention_cau, type_cau), \
               self.fuse_probability(pro_temp, mention_temp, type_temp), self.fuse_probability(pro_cof, mention_cof, type_cof)

    # 多token事件特殊标识符采用平均初始化
    def handler(self, to_add, tokenizer):
        da = self.RoBERTa_MLM.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp

        da = self.RoBERTa_MLM_for_schema.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
    def pro_adjacency(self, adjacency):
        # Delete relation between nodes to be predicted.
        adjacency = adjacency.float().to(device)
        temp = torch.where(adjacency == -1, torch.tensor(0, dtype=torch.float32).to(device), adjacency)
        # add self loop
        for i in range(temp.shape[0]):
            temp[i, torch.arange(temp.shape[1]), torch.arange(temp.shape[1])] = 1
        return temp

    def fuse_probability(self, a, b, c):
        return (1 - self.mention_ratio - self.type_ratio) * F.softmax(a, dim=1) + self.mention_ratio * F.softmax(b, dim=1) + self.type_ratio * F.softmax(c, dim=1)


