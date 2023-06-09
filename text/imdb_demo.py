import os

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

imdb_dir = '/Users/wangyao/tmp/aclImdb/'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

print("preparing text...")
# 将数据导入到设定好的容器内
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)


# 对数据进行分词
maxlen = 100  # 在100个单词后截断评论
training_samples = 200  # 在200个样本上训练
validation_samples = 10000  # 在10000个样本上验证
max_words = 10000  # 只考虑数据集中前10000个最常见的单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 将数据划分为训练集和验证集，打乱顺序，将正面评论和负面评论都混合在一起
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
