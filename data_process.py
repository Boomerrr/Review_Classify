import pickle

import nltk
import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 加载 spacy 模型
nlp = spacy.load("en_core_web_lg")

label_map = {'bug report': 0, 'irrelevant information': 1, 'feature request': 2}

# 定义函数来加载数据
def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            content, label = line.strip().split('\t')
            texts.append(content)
            labels.append(label_map[label])
    return texts, labels

# 定义递归函数来构建依存关系树
def build_tree(token):
    tree = f"{token.dep_} {token.text}"
    for child in token.children:
        tree += f" {build_tree(child)}"
    tree += ""
    return tree

# 加载训练集和测试集数据
train_file_path = './data/llama3.txt'
test_file_path = './data/test_filter.txt'
# train_texts, train_labels = load_data(train_file_path)
test_texts, test_labels = load_data(test_file_path)

# train_texts, valid_texts, train_labels, valid_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=3407)
# print(f'len train_texts : {len(train_texts)}')
# print(f'len valid_texts : {len(valid_texts)}')
print(f'len test_texts : {len(test_texts)}')

# 对训练集数据进行词性分析和依存树构建
# train_tokenized_data = []
# for sent, label in tqdm(zip(train_texts, train_labels), desc="Tokenizing Train Sentences"):
#    tokenized_sentence = [(token.text, token.pos_) for token in nlp(sent)]
#    doc = nlp(sent)
#    tree = None
#    # 循环只有一次
#    for sentence in doc.sents:
#        root = [token for token in sentence if token.head == token][0]
#        tree = build_tree(root)
#    train_tokenized_data.append((sent, tokenized_sentence, tree,label))
#
# valid_tokenized_data = []
# for sent, label in tqdm(zip(valid_texts, valid_labels), desc="Tokenizing Valid Sentences"):
#    tokenized_sentence = [(token.text, token.pos_) for token in nlp(sent)]
#    doc = nlp(sent)
#    tree = None
#    for sentence in doc.sents:
#        root = [token for token in sentence if token.head == token][0]
#        tree = build_tree(root)
#    valid_tokenized_data.append((sent, tokenized_sentence, tree,label))
#
#
test_tokenized_data = []
for sent, label in tqdm(zip(test_texts, test_labels), desc="Tokenizing Test Sentences"):
    tokenized_sentence = [(token.text, token.pos_) for token in nlp(sent)]
    doc = nlp(sent)
    tree = None
    for sentence in doc.sents:
        root = [token for token in sentence if token.head == token][0]
        tree = build_tree(root)
    test_tokenized_data.append((sent, tokenized_sentence, tree,label))

# 打印部分处理结果以确认
#print("Sample train data:", train_tokenized_data[:1])
#print("Sample valid data:", valid_tokenized_data[:1])
print("Sample test data:", test_tokenized_data[:1])

# 将列表写入.pkl文件
# with open('./data/train_classify_data.pkl', 'wb') as f:
#    pickle.dump(train_tokenized_data, f)
#
# with open('./data/valid_classify_data.pkl', 'wb') as f:
#    pickle.dump(valid_tokenized_data, f)

with open('./data/test_filter_classify_data.pkl', 'wb') as f:
    pickle.dump(test_tokenized_data, f)
