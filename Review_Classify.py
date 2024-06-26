import argparse
import pickle
import random
import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW,RobertaModel,RobertaTokenizer
import torch.nn as nn
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
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stop.")
    parser.add_argument("--architecture", type=str, default="Review_Classify", help="Architecture")
    parser.add_argument("--hidden_size", type=int, default=768, help="The size of hidden size.")
    parser.add_argument("--weight_decay", type=int, default=1e-4, help="The size of hidden size.")
    parser.add_argument("--use_wandb", type=str, default=False, help="Weather use wandb")
    args = parser.parse_args()
    return args

args = parse_args()

use_wandb = args.use_wandb
if use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="review_classify",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.learning_rate,
        "architecture": args.architecture,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed":args.seed,
        "num_labels":args.num_labels,
        "batch_size":args.batch_size,
        "max_seq_len":args.max_seq_len,
        "patience":args.patience,
        "model_name":args.model_name,
        "hidden_szie": args.hidden_size,
        "add_hidden_size": args.add_hidden_size,
        "weight_decay" : args.weight_decay,
        }
    )
    print(wandb.config)

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
weight_decay = args.weight_decay
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
print("hidden_size",hidden_size)
print("weight_decay",weight_decay)
print("-------------------------------------------------")


# 固定随机数
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


with open('./data/train_classify_data.pkl', 'rb') as f:
    train_tokenized_data = pickle.load(f)

with open('./data/valid_classify_data.pkl', 'rb') as f:
    valid_tokenized_data = pickle.load(f)

with open('./data/test_filter_classify_data.pkl', 'rb') as f:
    test_tokenized_data = pickle.load(f)

#train_tokenized_data = train_tokenized_data[:50]
#valid_tokenized_data = valid_tokenized_data[:50]
#test_tokenized_data = test_tokenized_data[:50]

# Tokenizer from transformers
if model=='ROBERTA_CLS':
    model_path = './roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = Feature_Aug_Classify(model,model_path=model_path,hidden_size = hidden_size,head_num = 12,pos = True,content = True,dependency = True)
elif model=='BERT_CLS':
    model_path = './bert_base'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = Feature_Aug_Classify(model,model_path=model_path,hidden_size = hidden_size,head_num = 12,pos = True ,content = True,dependency = True)

# Create dataset and data loader
train_dataset = Review_Data_Dataset(train_tokenized_data,tokenizer)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size)

valid_dataset = Review_Data_Dataset(valid_tokenized_data,tokenizer)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

test_dataset = Review_Data_Dataset(test_tokenized_data,tokenizer)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

print("Data prepared ! ")

model.to('cuda')
# Use PyTorch's AdamW instead of the one from transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

best_model=None
patience = args.patience
best_valid_f1 = 0.0
early_stopping_counter = 0
# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_data_loader, desc="Training (Epoch {})".format(epoch + 1)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        input_ids_pos = batch['input_ids_pos'].to('cuda')
        attention_mask_pos = batch['attention_mask_pos'].to('cuda')
        input_ids_dep = batch['input_ids_dep'].to('cuda')
        attention_mask_dep = batch['attention_mask_dep'].to('cuda')
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
        labels = batch['labels'].to('cuda')
        outputs = model(content_input, pos_input,dependency_input)
        loss = nn.CrossEntropyLoss()(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input_ids.size(0)
    current_loss = train_loss / len(train_data_loader)
    print("Epoch {} Loss: {}".format(epoch + 1, current_loss))
    # Validation
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch_valid in tqdm(valid_data_loader, desc="Validating..."):
            input_ids = batch_valid['input_ids'].to('cuda')
            attention_mask = batch_valid['attention_mask'].to('cuda')
            input_ids_pos = batch_valid['input_ids_pos'].to('cuda')
            attention_mask_pos = batch_valid['attention_mask_pos'].to('cuda')
            input_ids_dep = batch_valid['input_ids_dep'].to('cuda')
            attention_mask_dep = batch_valid['attention_mask_dep'].to('cuda')
            labels = batch_valid['labels'].to('cuda')
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
            outputs = model(content_input,pos_input,dependency_input)
            _, predicted = torch.max(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    valid_f1 = f1_score(y_true, y_pred, average="macro")
    print("Epoch {} Validation F1: {:.2f}%".format(epoch + 1, valid_f1 * 100))
    if use_wandb:
        wandb.log({"valid_f1": valid_f1, "loss": current_loss})
    if valid_f1 > best_valid_f1:
        best_valid_f1 = valid_f1
        early_stopping_counter = 0
        best_model = model.state_dict()
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered! Stop training.,Current epoch is {epoch+1}")
            break
model.load_state_dict(best_model)
model.eval()

torch.save(model.state_dict(), './saved_model/1_1_1_review_classify_model_roberta.pth')

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
        outputs = model(content_input,pos_input,dependency_input)
        _,predicted = torch.max(outputs, dim=1)
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

with open('./saved_model/all_labels_111.pkl', 'wb') as f:
    pickle.dump(all_labels, f)
with open(f'./saved_model/1_1_1_roberta_predictions.pkl', 'wb') as f:
    pickle.dump(all_predictions, f)

if use_wandb:
    wandb.log({"Test Accuracy": accuracy, "Test Precision": precision,"Test Recall":recall,"Test F1 Score":f1})

