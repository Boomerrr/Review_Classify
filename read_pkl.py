import pickle

# 读取pkl文件
with open('./data/test_classify_data.pkl', 'rb') as f:
    features = pickle.load(f)

print(features[0])





