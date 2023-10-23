from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score
import pickle


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    F1 = fbeta_score(labels, pred_y, beta=1)
    return acc, precision, sensitivity, specificity, MCC, F1

def seq_to_num(line, seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[j] = protein_dict[line[j]]
    return seq

seq_length = 121  # The benchmark dataset was set to 121
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
protein_dict = dict((c, i) for i, c in enumerate(amino_acids))

# M1
M1_feature = pd.read_csv("./data/test dataset/firstStage/feature.csv", header=None)  # Amino acid features
M1_feature = np.array(pd.DataFrame(M1_feature))

# M2
file = open("./data/test dataset/firstStage/firstStage_test.faa", encoding="utf-8") # Sequences
all_line = file.readlines()
fasta = []
for i in range(len(all_line)):
    if i % 2 == 1:
        fasta.append(all_line[i][0:-1])
M2_feature = []
for i in range(len(fasta)):
    line = fasta[i]
    seq = seq_to_num(line, seq_length)
    M2_feature.append(seq)
M2_feature = np.array(M2_feature)

# label
label_pos = np.ones((691, 1))
label_neg = np.zeros((2522, 1))
label = np.append(label_pos, label_neg)

scaler = pickle.load(open("./model/firstStage/scaler_first.pkl", 'rb'))
[a, b] = np.shape(M1_feature)
M1_feature = scaler.transform(M1_feature).reshape(a, b, -1)
cv_clf = load_model("./model/firstStage/FirstStage_model.h5")  # model
preds = cv_clf.predict([M1_feature, M2_feature])

pred_y = np.rint(preds)
acc, precision, sensitivity, specificity, MCC, F1 = calculate_performace(len(label), pred_y, label)
fpr, tpr, _ = roc_curve(label, preds)
AUC = auc(fpr, tpr)
pre, rec, _ = precision_recall_curve(label, preds)
AUPR = auc(rec, pre)
print('acc=%f,precision=%f,sensitivity=%f,specificity=%f,MCC=%f,AUC=%f,AUPR=%f, F1=%f'
      % (acc, precision, sensitivity, specificity, MCC, AUC, AUPR, F1))
