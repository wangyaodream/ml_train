from keras.preprocessing.text import Tokenizer


samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 创建一个分词器，设定为只考虑前1000个词
tokenizer = Tokenizer(num_words=1000)
# 构建单词索引
tokenizer.fit_on_texts(samples)

# 将字符串转换为整数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index  # 找回单词索引
print('Found %s unique tokens.' % len(word_index))


