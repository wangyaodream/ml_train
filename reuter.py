
import numpy as np
from keras.datasets import reuters


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

