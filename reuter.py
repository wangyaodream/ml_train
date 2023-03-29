
import numpy as np
from keras.datasets import reuters
from keras import models, layers
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

# get dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# 准备数据
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# 训练数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 标签向量化

# 手动实现one-hot编码
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels, dimension)))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# 构建网络
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation="softmax"))  # softmax激活将结果转换成不同输出类别上的概率分布

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',  # 衡量两个概率分布之间的距离
              metrics=['accuracy'])


# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


# 开始训练
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 绘制训练损失和验证损失
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)  # 20

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# 绘制训练精度和验证精度
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
