from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
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

def plot_roc_curve(source):
    '''
    Plot auc curve from cross validation
    :param source: list of tuple of (y_pred, y_ture), source data for plot figures
    :param file_name: str, file to save figures
    :return: None
    '''
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (y_true, y_pred) in enumerate(source):
        # y_pred = y_pred[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.4f)' % (i, roc_auc))
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # plot diagonal
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Blind Guess', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_tpr[0] = 0.
    mean_auc = sum(aucs) / len(aucs)  # calculate mean_auc
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.4f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

def seq_to_num(line, seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[j] = protein_dict[line[j]]
    return seq

seq_length = 121  # The benchmark dataset was set to 121, Two independent data sets were set to 110
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

label_pos = np.ones((691, 1))
label_neg = np.zeros((2522, 1))
label = np.append(label_pos, label_neg)

auc_l = []
real_labels = []
for val in label:
    if val == 1:
        real_labels.append(1)
    else:
        real_labels.append(0)
auc_label = (real_labels,)

scaler = pickle.load(open("./model/firstStage/scaler_first.pkl", 'rb'))
[a, b] = np.shape(M1_feature)
M1_feature = scaler.transform(M1_feature).reshape(a, b, -1)
cv_clf = load_model("./model/firstStage/FirstStage_model.h5")  # model


preds = cv_clf.predict([M1_feature, M2_feature])
lstm_class = np.rint(preds)
acc, precision, sensitivity, specificity, MCC, F1 = calculate_performace(len(label), lstm_class, label)
fpr, tpr, _ = roc_curve(label, preds)
roc_auc = auc(fpr, tpr)
pre, rec, _ = precision_recall_curve(label, preds)
aucpr = auc(rec, pre)
print('GTB:acc=%f,precision=%f,sensitivity=%f,specificity=%f,MCC=%f,roc=%f,aucpr=%f, F1=%f'
      % (acc, precision, sensitivity, specificity, MCC, roc_auc, aucpr,F1))
auc_pred = (preds,)
auc_all = auc_label + auc_pred
auc_l.append(auc_all)
plot_roc_curve(auc_l)


