import argparse
import pickle
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, RobertaModel, RobertaTokenizer
from model import Feature_Aug_Classify, Review_Data_Dataset


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model_name", type=str, default="ROBERTA_CLS", help="The path of used model.")
    parser.add_argument("--dataset", type=str, default="llama3", help="The dataset to use, ATIS or SNIPS.")
    parser.add_argument("--seed", type=int, default=3407, help="The random seed.")
    parser.add_argument("--num_labels", type=int, default=3, help="The number of dataset label.")
    parser.add_argument("--epochs", type=int, default=50, help="The number of iterations for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for train and validation.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="The max length of sequence.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stop.")
    parser.add_argument("--architecture", type=str, default="Review_Classify", help="Architecture")
    parser.add_argument("--hidden_size", type=int, default=768, help="The size of hidden size.")
    parser.add_argument("--use_wandb", type=str, default=False, help="Weather use wandb")
    args = parser.parse_args()
    return args


args = parse_args()

# 参数传递
dataset = args.dataset
seed = args.seed
num_labels = args.num_labels
lr = args.learning_rate
batch_size = args.batch_size
max_length = args.max_seq_len
epochs = args.epochs
model = args.model_name
patience = args.patience
hidden_size = args.hidden_size
print("-------------------------------------------------")
print("The args for train")
print("dataset:", dataset)
print("seed:", seed)
print("model:", model)
print("num_labels:", num_labels)
print("lr:", lr)
print("batch_size:", batch_size)
print("max_length:", max_length)
print("epochs:", epochs)
print("patience:", patience)
print("hidden_size", args.hidden_size)
print("-------------------------------------------------")

# 固定随机数
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

with open('../data/test_filter_classify_data.pkl', 'rb') as f:
    test_tokenized_data = pickle.load(f)
# test_tokenized_data = train_tokenized_data[:50]


# Tokenizer from transformers
if model == 'ROBERTA_CLS':
    model_path = '../model/roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = Feature_Aug_Classify(model, model_path=model_path, hidden_size=hidden_size, head_num=12, pos=True,
                                 content=False, dependency=True)
elif model == 'BERT_CLS':
    model_path = '../model/bert_base'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = Feature_Aug_Classify(model, model_path=model_path, hidden_size=hidden_size, head_num=12, pos=True,
                                 content=True, dependency=True)

test_dataset = Review_Data_Dataset(test_tokenized_data, tokenizer)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

print("Data prepared ! ")

model.to('cuda')

file_path = '../saved_model/0_1_1_review_classify_model_roberta.pth'
model.load_state_dict(torch.load(file_path))

model.eval()

all_labels = []
all_predictions = []
with torch.no_grad():
    for batch in tqdm(test_data_loader, desc="Testing"):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        input_ids_pos = batch['input_ids_pos'].to('cuda')
        attention_mask_pos = batch['attention_mask_pos'].to('cuda')
        input_ids_dep = batch['input_ids_dep'].to('cuda')
        attention_mask_dep = batch['attention_mask_dep'].to('cuda')
        labels = batch['labels'].to('cuda')
        content_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        pos_input = {
            'input_ids': input_ids_pos,
            'attention_mask': attention_mask_pos
        }
        dependency_input = {
            'input_ids': input_ids_dep,
            'attention_mask': attention_mask_dep
        }
        outputs = model(content_input, pos_input, dependency_input)
        _, predicted = torch.max(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Test Precision: {:.2f}%".format(precision * 100))
print("Test Recall: {:.2f}%".format(recall * 100))
print("Test F1 Score: {:.2f}%".format(f1 * 100))
