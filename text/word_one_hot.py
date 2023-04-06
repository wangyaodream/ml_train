import numpy as np


samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_lenght = 10  # 对样本进行分词时只考虑前max_lenght个单词

results = np.zeros(shape=(len(samples),
                         max_lenght,
                         max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_lenght]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print(results.shape)
