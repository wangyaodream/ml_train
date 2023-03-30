import json

from keras.datasets import boston_housing
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 预处理数据，将数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) 
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model

# 进行k折验证, 目的是可靠的评估模型
k = 4  # 折4次
num_val_samples = len(train_data) // k  # 每一折所包含的数量
num_epochs = 500
# all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    # 准备训练数据：第k个分区的数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 准备训练数据
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
    model = build_model()
    history = model.fit(partial_train_data,
                      partial_train_targets,
                      epochs=num_epochs,
                      batch_size=1,
                      verbose=0)
    # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    # all_scores.append(val_mae)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
with open('temp/result.json', "w") as fp:
    fp.write(json.dumps(all_mae_histories))

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]
# print(all_scores)
# print(np.mean(all_scores))

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

