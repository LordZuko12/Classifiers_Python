import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filename = "tweet.txt"
text = (open(filename).read()).lower()
unique_chars = sorted(list(set(text)))

char_to_int = {}
int_to_char = {}

for i, c in enumerate (unique_chars):
    char_to_int.update({c: i})
    int_to_char.update({i: c})

x = []
y = []

for i in range(0, len(text) - 50, 1):
    sequence = text[i:i + 50]
    label =text[i + 50]
    x.append([char_to_int[char] for char in sequence])
    y.append(char_to_int[label])

x_modified = numpy.reshape(x, (len(x), 50, 1))
x_modified = x_modified / float(len(unique_chars))
y_modified = np_utils.to_categorical(y)

model = Sequential()
model.add(LSTM(300, input_shape=(x_modified.shape[1], x_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x_modified, y_modified, epochs=1, batch_size=30)
start_index = numpy.random.randint(0, len(x)-1)
new_string = x[start_index]


for i in range(50):
    x1 = numpy.reshape(new_string, (1, len(new_string), 1))
    x1 = x1 / float(len(unique_chars))

    pred_index = numpy.argmax(model.predict(x1, verbose=0))
    char_out = int_to_char[pred_index]
    seq_in = [int_to_char[value] for value in new_string]
    print(char_out)

    new_string.append(pred_index)
    new_string = new_string[1:len(new_string)]

