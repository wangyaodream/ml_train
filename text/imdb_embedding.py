from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.datasets import imdb
from keras.utils import pad_sequences


max_feature = 10000
maxlen = 50


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)


