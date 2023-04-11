import os

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import matplotlib.pyplot as plt

imdb_dir = '/Users/wangyao/tmp/aclImdb/'
glove_dir = '/Users/wangyao/tmp/glove.6B/'
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

# 将数据划分为训练集和验证集，需要打乱数据
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
# 将数据以新的顺序进行装配
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]  # [:200]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]  # [200: 200 + 10000]
y_val = labels[training_samples: training_samples + validation_samples]

# 解析GloVe词嵌入文件
embedding_index = {}
with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as fp:
    for line in fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs


# 准备GloVe词嵌入矩阵
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
         embedding_vector = embedding_index.get(word)
         if embedding_vector is not None:
             embedding_matrix[i] = embedding_vector

# 定义模型
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
summary = model.summary()

# 将预训练的词嵌入加载到Embedding层中
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# 编译模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('../temp/pre_trained_glove_model.h5')

# 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accruracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
