#import tensorflow as tf
import numpy as np
import os
import json
import math
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

json_data={}
with open(os.path.join('en_medical_dialog.json'), 'r') as file:
                data = json.load(file)
                #json_data['data'] = data
                #print(data)

# Sample data
doctor_sentences_train = []
patient_sentences_train = []
doctor_yAxis = []
patient_yAxis = []

doctor_sentences_test = []
patient_sentences_test = []
doctor_yAxis_test = []
patient_yAxis_test = []


# Split data into training and testing sets

trainingData = data[0: math.floor(len(data)*0.8)]
testingData = data[math.floor(len(data)*0.8)+1:len(data)-1]
print(testingData[-1])

for i in trainingData:
  doctor_sentences_train.append(i['Doctor'])
  doctor_yAxis.append(1)
  patient_sentences_train.append(i['Patient'])
  patient_yAxis.append(0)

for i in testingData:
  doctor_sentences_test.append(i['Doctor'])
  doctor_yAxis_test.append(1)
  patient_sentences_test.append(i['Patient'])
  patient_yAxis_test.append(0)


testDataSet = doctor_sentences_test + patient_sentences_test

print(len(doctor_sentences_train))
print(len(patient_sentences_train))

# Labels (1 for doctor, 0 for patient)
labels = doctor_yAxis + patient_yAxis

labelsTest = doctor_yAxis_test + patient_yAxis_test


print(len(labels))


# Combine sentences and labels
all_sentences = doctor_sentences_train + patient_sentences_train
all_labels = np.array(labels)

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(all_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(all_sentences)
padded_sequences = pad_sequences(sequences, maxlen=50, truncating='post', padding='post')

# Model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 16, input_length=50))
model.add(LSTM(60, dropout=0.2))
model.add(Dense(60, activation='relu', kernel_regularizer=regularizers.l2(0.5)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(padded_sequences, all_labels, epochs=1, validation_split=0.4)

# Test with a new sentence

new_sentence = testDataSet #["Well! I see the temperature is high but lets observe for 2 more days. I will prescribe papasitomal for now, "]
new_sequence = tokenizer.texts_to_sequences(new_sentence)
new_padded_sequence = pad_sequences(new_sequence, maxlen=50, truncating='post', padding='post')
prediction = model.predict(new_padded_sequence)

# Output the prediction
print("Prediction:", prediction[math.floor(len(prediction)/2):len(prediction)-1])

# Using a lambda function and map to update values
resultLabels = list(map(lambda x: "Doctor" if x > 0.5 else "Patient", prediction))
expectedLabels = list(map(lambda x: "Doctor" if x == 1 else "Patient", labelsTest))

# Define a lambda function to create a tuple from elements of three lists at the same index
create_tuple = lambda x, y, z: (x, y, z)
result = list(map(create_tuple, testDataSet, resultLabels, expectedLabels))
print(result[-1])

import pandas as pd
df = pd.DataFrame(result, columns=['Statement', 'Predicted', 'Expected'])

# Export the DataFrame to an Excel file
df.to_excel('output.xlsx', index=False)