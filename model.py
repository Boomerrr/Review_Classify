import math

import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel, RobertaModel
import sys
import os


class Review_Data_Dataset(Dataset):
    def __init__(self, tokenized_data, tokenizer):
        # (sent, tokenized_sentence, tree,label)
        self.encodings_word = [
            tokenizer(sentence[0], truncation=True, padding='max_length', max_length=128, return_tensors="pt") for
            sentence in tqdm(tokenized_data, desc="Encoding Words")
        ]
        self.encodings_pos = [
            tokenizer(" ".join([token[0] for token in sentence[1]]), truncation=True, padding='max_length',
                      max_length=128, return_tensors="pt")
            for sentence in tqdm(tokenized_data, desc="Encoding POS")
        ]
        self.encodings_dependency = [
            tokenizer(" ".join([token for token in sentence[2]]), truncation=True, padding='max_length', max_length=128,
                      return_tensors="pt")
            for sentence in tqdm(tokenized_data, desc="Encoding Dependencies")
        ]
        self.labels = [sentence[3] for sentence in tokenized_data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: self.encodings_word[idx][key].squeeze(0) for key in self.encodings_pos[idx]}
        item['input_ids_pos'] = self.encodings_pos[idx]['input_ids'].squeeze(0)
        item['attention_mask_pos'] = self.encodings_pos[idx]['attention_mask'].squeeze(0)
        item['input_ids_dep'] = self.encodings_dependency[idx]['input_ids'].squeeze(0)
        item['attention_mask_dep'] = self.encodings_dependency[idx]['attention_mask'].squeeze(0)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        # for key, value in item.items():
        #    print(f"{key}: {value.shape}")
        return item


class Reiveiw_Classify(nn.Module):
    def __init__(self, input_size):
        super(Reiveiw_Classify, self).__init__()
        self.fc1 = nn.Linear(input_size, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc1(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty([in_features, out_features], dtype=torch.float),
            requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty([out_features], dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, inputs, identity=False):
        if identity:
            return torch.matmul(adj, self.weight)
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Feature_Aug_Classify(nn.Module):
    def __init__(self, model, model_path, hidden_size, head_num, pos, content, dependency):
        super(Feature_Aug_Classify, self).__init__()
        if model == 'BERT_CLS':
            self.bert_content = BertModel.from_pretrained(model_path)
            self.bert_pos = BertModel.from_pretrained(model_path)
            self.bert_dependency = BertModel.from_pretrained(model_path)
        else:
            self.bert_content = RobertaModel.from_pretrained(model_path)
            self.bert_pos = RobertaModel.from_pretrained(model_path)
            self.bert_dependency = RobertaModel.from_pretrained(model_path)
        self.head_num = head_num
        self.gcns_content = nn.ModuleList([GCN(768, hidden_size, True) for _ in range(head_num)])
        self.gcns_pos = nn.ModuleList([GCN(768, hidden_size, True) for _ in range(head_num)])
        self.gcns_dependency = nn.ModuleList([GCN(768, hidden_size, True) for _ in range(head_num)])

        count = sum([pos, content, dependency])
        self.classify = Reiveiw_Classify(768 * count)
        self.pos = pos
        self.content = content
        self.dependency = dependency
        self.drop = nn.Dropout(0.1)

    def forward(self, content_input, pos_input, dependency_input):
        feature_content = None
        feature_pos = None
        feature_dependency = None
        # Word
        if self.content:
            content_output = self.bert_content(**content_input, output_attentions=True)
            content_attention = content_output.attentions
            content_hidden = content_output.last_hidden_state
            feature_content = self.process_attention(content_attention, content_hidden, self.gcns_content,
                                                     content_input['attention_mask'])
        # Pos
        if self.pos:
            pos_output = self.bert_pos(**pos_input, output_attentions=True)
            pos_attention = pos_output.attentions
            pos_hidden = pos_output.last_hidden_state
            feature_pos = self.process_attention(pos_attention, pos_hidden, self.gcns_pos, pos_input['attention_mask'])

        # Dependency
        if self.dependency:
            dependency_output = self.bert_dependency(**dependency_input, output_attentions=True)
            dependency_attention = dependency_output.attentions
            dependency_hidden = dependency_output.last_hidden_state
            feature_dependency = self.process_attention(dependency_attention, dependency_hidden, self.gcns_dependency,
                                                        dependency_input['attention_mask'])

        output = torch.cat((feature_content, feature_pos, feature_dependency), dim=1)
        # output = torch.cat((feature_pos,feature_dependency), dim=1)
        # print(output.shape)
        logit = self.classify(output)
        return logit

    def process_attention(self, attentions, hidden_state, gcns, atten_mask):
        k = 3
        attentions = attentions[-1]
        # print(f'attentions.shape : {attentions.shape}')
        out_heads = []
        # Multi head attention matrix GCN
        for head in range(self.head_num):
            out = gcns[head].forward(attentions[:, head, :, :], hidden_state)
            out_heads.append(out)
        output = torch.stack(out_heads, dim=1)
        topk_indices = torch.topk(output, k=k, dim=1)[1]
        topk_values = torch.gather(output, 3, topk_indices)
        out = torch.mean(topk_values, dim=1)

        # Pool token layer
        attention_mask_expanded = atten_mask.unsqueeze(-1).expand(out.size()).float()
        sum_embeddings = torch.sum(out * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        avg_hidden_state = sum_embeddings / sum_mask

        return avg_hidden_state
