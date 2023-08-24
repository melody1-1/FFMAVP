from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score
from sklearn import metrics
from tensorflow.keras import backend as K
from numpy import *
from keras.models import Sequential, load_model
from keras import backend as K
import numpy as np
import pickle


def seq_to_num(line, seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[j] = protein_dict[line[j]]
    return seq

def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def calculate_performace(label, pred_y):
    for i in range(len(pred_y)):
        max_value = max(pred_y[i])
        for j in range(len(pred_y[i])):
            if max_value == pred_y[i][j]:
                pred_y[i][j] = 1
            else:
                pred_y[i][j] = 0
    precision = precision_score(label, pred_y, average='macro')
    recall = recall_score(label, pred_y, average='macro')
    f1 = f1_score(label, pred_y, average='macro')
    acc = metrics.accuracy_score(label, pred_y)
    return acc, precision, recall, f1

def categorical_focal_loss_1(gamma):
    def categorical_focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=-1))
    return categorical_focal_loss_fixed

seq_length = 121  # The benchmark dataset was set to 121
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
protein_dict = dict((c, i) for i, c in enumerate(amino_acids))

# label_family
label_family = pd.read_csv("./data/test dataset/secondStage/label_family.csv", header=None)
label_family = np.array(pd.DataFrame(label_family))

# label_virus
label_virus = pd.read_csv("./data/test dataset/secondStage/label_virus.csv", header=None)
label_virus = np.array(pd.DataFrame(label_virus))

# M1
M1_feature = pd.read_csv("./data/test dataset/secondStage/feature.csv", header=None)
M1_feature = np.array(pd.DataFrame(M1_feature))

# M2
file = open("./data/test dataset/secondStage/secondStage_test.faa", encoding="utf-8")
all_line = file.readlines()
fasta = []
for i in range(len(all_line)):
    if (i % 2 == 1):
        fasta.append(all_line[i][0:-1])
M2_feature = []
for i in range(len(fasta)):
    line = fasta[i]
    M2_feature.append(seq_to_num(line, seq_length))
M2_feature = np.array(M2_feature)

encoder = LabelEncoder()
label_family = encoder.fit_transform(label_family)
label_virus = encoder.fit_transform(label_virus)

# family data and label
x_test_family_M1 = []
x_test_family_M2 = []
y_test_family = []
for x, y in enumerate(label_family):
    if y != 6:
        x_test_family_M1.append(M1_feature[x])
        x_test_family_M2.append(M2_feature[x])
        y_test_family.append(label_family[x])
# features
x_test_family_M1 = np.array(x_test_family_M1)
x_test_family_M2 = np.array(x_test_family_M2)
[a1, b1] = np.shape(x_test_family_M1)
scaler = pickle.load(open("./model/secondStage/scaler_second.pkl", 'rb'))
x_test_family_M1 = scaler.transform(x_test_family_M1).reshape(a1, b1, -1)
# label_family
y_family = np.array(y_test_family)
y_family = to_one_hot(y_family, dimension=6)

# virus data and label
x_test_virus_M1 = []
x_test_virus_M2 = []
y_test_virus = []
for x, y in enumerate(label_virus):
    if y != 8:
        x_test_virus_M1.append(M1_feature[x])
        x_test_virus_M2.append(M2_feature[x])
        y_test_virus.append(label_virus[x])
# features
x_test_virus_M1 = np.array(x_test_virus_M1)
x_test_virus_M2 = np.array(x_test_virus_M2)
[a2, b2] = np.shape(x_test_virus_M1)
scaler = pickle.load(open("./model/secondStage/scaler_second.pkl", 'rb'))
x_test_virus_M1 = scaler.transform(x_test_virus_M1).reshape(a2, b2, -1)
# label_virus
y_virus = np.array(y_test_virus)
y_virus = to_one_hot(y_virus, dimension=8)


cv_clf = load_model("./model/secondStage/SecondStage_model.h5",
                    custom_objects={'categorical_focal_loss_fixed': categorical_focal_loss_1(gamma=0)})

# ---------------------------------Task 1 Prediction-------------------------------------------
preds_family = cv_clf.predict([x_test_family_M1, x_test_family_M2])
preds_family = preds_family[0][:, :6]
print("******************************task1********************************")
acc, precision, recall, f1 = calculate_performace(y_family, preds_family)
print('GTB:acc=%f,precision=%f,recall=%f,f1_score=%f' % (acc, precision, recall, f1))

# ---------------------------------Task 2 Prediction-------------------------------------------
preds_virus = cv_clf.predict([x_test_virus_M1, x_test_virus_M2])
preds_virus = preds_virus[1][:, :8]
acc, precision, recall, f1 = calculate_performace(y_virus, preds_virus)
print("******************************task2********************************")
print('GTB:acc=%f,precision=%f,recall=%f,f1_score=%f' % (acc, precision, recall, f1))








