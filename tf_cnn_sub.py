####
# Similar to tf_nn_classifier.py, just that the model trains on all
# 9000 training examples for 25 epochs and makes a prediction on the test data for submission
####
import numpy as np
from sklearn import preprocessing
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import optimizers, regularizers

cale = "F:/Projects/IA/SmartphoneUserClassification/data/"

train_labels = []
ids = []
with open(cale + "train_labels.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if line_count >= 10000:
                break
            ids.append(row[0])
            train_labels.append(row[1])
            line_count += 1
    print(f'Processed {line_count} lines.')

train_data = np.zeros((len(ids), 139, 3))

for i in range(len(ids)):
    with open(cale + "train/" + ids[i] + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        print(i)
        line = 0
        for row in csv_reader:
            if line == 139:
                break
            train_data[i, line, 0] = row[0]
            train_data[i, line, 1] = row[1]
            train_data[i, line, 2] = row[2]
            line += 1

train_data = train_data.astype(float)

test_ids = []
with open(cale + "sample_submission.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if line_count >= 10000:
                break
            test_ids.append(row[0])
            line_count += 1
    print(f'Processed {line_count} lines.')

test_data = np.zeros((len(test_ids), 139, 3))

for i in range(len(test_ids)):
    with open(cale + "test/" + test_ids[i] + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        print(i)
        line = 0
        for row in csv_reader:
            if line == 139:
                break
            test_data[i, line, 0] = row[0]
            test_data[i, line, 1] = row[1]
            test_data[i, line, 2] = row[2]
            line += 1

test_data = test_data.astype(float)

train_labels = np.asarray(train_labels)
train_labels = train_labels.astype(float)
for i in range(train_labels.shape[0]):
    train_labels[i] -= 1

train_shape0 = train_data.shape[0]
test_shape0 = test_data.shape[0]

scaler = preprocessing.StandardScaler()
scaler.fit(train_data.reshape([train_shape0, 139 * 3]))
train_scaled = scaler.transform(train_data.reshape([train_shape0, 139 * 3]))
test_scaled = scaler.transform(test_data.reshape([test_shape0, 139 * 3]))
train_data = train_scaled.reshape([train_shape0, 139, 3])
test_data = test_scaled.reshape([test_shape0, 139, 3])

initializer = keras.initializers.glorot_normal(seed=None)

model = keras.Sequential([
    keras.layers.Conv1D(100, 10, activation='relu', input_shape=(139, 3)),
    keras.layers.Conv1D(100, 10, activation='relu'),
    keras.layers.MaxPooling1D(3),
    keras.layers.Conv1D(160, 10, activation='relu'),
    keras.layers.Conv1D(160, 10, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer=initializer,
                       kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=initializer,
                       kernel_regularizer=regularizers.l1(0.01), activity_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(20, activation=tf.nn.softmax)
])

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=25)
predictions = model.predict(test_data)

with open(cale + 'subs/' + 'tf_cnn_submission_25_epochs.csv', mode='w', newline='') as sub_file:
    writer = csv.writer(sub_file, delimiter=',')
    writer.writerow(['id', 'class'])

    for i in range(len(test_ids)):
        writer.writerow([test_ids[i], np.argmax(predictions[i]) + 1])
