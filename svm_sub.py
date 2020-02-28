#########
# Essentially the same script as svm.py, with the difference that the data is not split
# for validation, the model is trained on all 9000 examples, and a prediction is made
# on the test data for submission
#########
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv

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
            ids.append(row[0])
            train_labels.append(row[1])
            line_count += 1

train_data = np.zeros((len(ids), 150 * 3))

for i in range(len(ids)):
    with open(cale + "train/" + ids[i] + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        idx = 0
        for row in csv_reader:
            if idx == 450:
                break
            train_data[i, idx] = row[0]
            idx += 1
            train_data[i, idx] = row[1]
            idx += 1
            train_data[i, idx] = row[2]
            idx += 1

test_ids = []
with open(cale + "sample_submission.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            test_ids.append(row[0])
            line_count += 1

test_data = np.zeros((len(test_ids), 150 * 3))

for i in range(len(test_ids)):
    with open(cale + "test/" + test_ids[i] + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        idx = 0
        for row in csv_reader:
            if idx == 450:
                break
            test_data[i, idx] = row[0]
            idx += 1
            test_data[i, idx] = row[1]
            idx += 1
            test_data[i, idx] = row[2]
            idx += 1


def normalize(norm_type, train_data, test_data):
    if norm_type == 'standard':
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    elif norm_type == 'minmax':
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    elif norm_type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    elif norm_type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data
    else:
        print('Not a normalization method')
        return


train_data = train_data.astype(float)
test_data = test_data.astype(float)
train_data, test_data = normalize('standard', train_data, test_data)

clasificator = SVC(C=100, kernel='rbf', degree=2, gamma='scale',
                   coef0=0.0, shrinking=True, probability=False,
                   tol=1e-3, cache_size=200, class_weight=None,
                   verbose=False, max_iter=-1, decision_function_shape='ovr',
                   random_state=None)
clasificator.fit(train_data, train_labels)
pred_labels = clasificator.predict(test_data)

with open(cale + 'subs/' + 'svm_submission_1.csv', mode='w', newline='') as sub_file:
    writer = csv.writer(sub_file, delimiter=',')
    writer.writerow(['id', 'class'])

    for i in range(len(test_ids)):
        writer.writerow([test_ids[i], pred_labels[i]])
