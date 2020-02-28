from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv

# Data loading
cale = "F:/Projects/IA/SmartphoneUserClassification/data/"
# train_labels = {}
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
    print(f'Processed {line_count} lines.')

#####
# Initially tried storing the data in a dictionary, but didn't follow up
#####
# train_data = {}

# for id, label in train_labels.items():
#     with open(cale + "train/" + id + ".csv") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         row_count = sum(1 for row in csv_reader)
#     with open(cale + "train/" + id + ".csv") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         train_data[id] = np.zeros((row_count, 3))
#
#         line = 0
#         for row in csv_reader:
#             train_data[id][line, 0] = row[0]
#             train_data[id][line, 1] = row[1]
#             train_data[id][line, 2] = row[2]
#             line += 1
#
# for id, data in train_data.items():
#     print(f'id: {id}')
#     for i in range(data.shape[0]):
#         print(f'\t{data[i, 0]}, {data[i, 1]}, {data[i, 2]}')
#####

train_data = np.zeros((len(ids), 139, 3))

for i in range(len(ids)):
    with open(cale + "train/" + ids[i] + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line = 0
        for row in csv_reader:
            if line == 139:
                break
            train_data[i, line, 0] = row[0]
            train_data[i, line, 1] = row[1]
            train_data[i, line, 2] = row[2]
            line += 1

train_data = train_data.astype(float)


def evaluate(pred_labels, real_labels):
    aux = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == real_labels[i]:
            aux += 1
    return aux / len(real_labels)


# Function used during first grid search for finding best normalization method
# - no longer necessary
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


# Data splitting for validation(ratio: 1/3 for validation)
split_train_data = train_data[0:6000]
split_test_data = train_data[6000:]
split_train_labels = train_labels[0:6000]
split_test_labels = train_labels[6000:]

# necessary conversions
split_train_labels = np.asarray(split_train_labels)
split_test_labels = np.asarray(split_test_labels)
split_train_labels = split_train_labels.astype(float)
split_test_labels = split_test_labels.astype(float)

# Data standardization
scaler = preprocessing.StandardScaler()
scaler.fit(split_train_data.reshape([6000, 139 * 3]))
train_scaled = scaler.transform(split_train_data.reshape([6000, 139 * 3]))
test_shape = split_test_data.shape[0]
test_scaled = scaler.transform(split_test_data.reshape([test_shape, 139 * 3]))
split_train_data = train_scaled.reshape([6000, 139 * 3])
split_test_data = test_scaled.reshape([test_shape, 139 * 3])


#######
# Parameter testing

split_train_copy, split_test_copy = split_train_data.copy(), split_test_data.copy()

# Manual Grid Search
# Obs.: Standard normalization method proved to be by far most efficient
#       from the start, so it was no longer included in subsequent grid searches
norm_type = ['standard', 'minmax', 'l1', 'l2']
Cs = [0.1, 0.3, 0.5, 0.7, 1]
kerns = ['rbf', 'poly']
accuracies_train = np.zeros((len(Cs), len(kerns)))
accuracies_test = np.zeros((len(Cs), len(kerns)))

for Cs_idx, Cs_val in enumerate(Cs):
    for kern_idx, kern_val in enumerate(kerns):

        clasificator = SVC(C=Cs_val, kernel=kern_val, degree=3, gamma='scale',
                           coef0=0.0, shrinking=True, probability=False,
                           tol=1e-3, cache_size=200, class_weight=None,
                           verbose=False, max_iter=-1, decision_function_shape='ovr',
                           random_state=None)
        clasificator.fit(split_train_data, split_train_labels)
        etichete_pred_train = clasificator.predict(split_train_data)
        etichete_pred_test = clasificator.predict(split_test_data)

        accuracies_test[Cs_idx, kern_idx] = evaluate(etichete_pred_test, split_test_labels)
        accuracies_train[Cs_idx, kern_idx] = evaluate(etichete_pred_train, split_train_labels)

for Cs_idx, Cs_val in enumerate(Cs):
    for kern_idx, kern_val in enumerate(kerns):
        print('norm_val, Cs_val, kern_val')
        print(Cs_val, kern_val)
        print(f'train: {accuracies_train[Cs_idx, kern_idx]}')
        print(f'test: {accuracies_test[Cs_idx, kern_idx]}')

for i in range(len(ids)):
    print(f'id: {ids[i]}')
    print(f'\t{train_data[i][:]}')


# Automated Grid Search with 3-fold cross validation using sklearn.model_selection library
param_grid = {'C': [1000], 'gamma': [0.001], 'kernel': ['rbf'],
              'degree': [2]}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(split_train_data, split_train_labels)

print(grid.best_params_)
print("train score - " + str(grid.score(split_train_data, split_train_labels)))
print("test score - " + str(grid.score(split_test_data, split_test_labels)))

pred = grid.predict(split_test_data)

print(evaluate(pred, split_test_labels))

# Defining a classifier using the best parameters found
clasificator = SVC(C=10, kernel='rbf', degree=2, gamma='scale',
                   coef0=0.0, shrinking=True, probability=False,
                   tol=1e-3, cache_size=200, class_weight=None,
                   verbose=False, max_iter=-1, decision_function_shape='ovr',
                   random_state=None)
clasificator.fit(split_train_data, split_train_labels)
etichete_pred_train = clasificator.predict(split_train_data)
etichete_pred_test = clasificator.predict(split_test_data)

print(evaluate(etichete_pred_test, split_test_labels))
