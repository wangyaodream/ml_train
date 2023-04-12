import os

from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import models


test_dir = "/Users/wangyao/tmp/aclImdb/test/"

labels = []
texts = []

for label_type in ["neg", "pos"]:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == ".txt":
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

tokenizer = Tokenizer()
sequences = Tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=100)
