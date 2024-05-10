# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:30:08 2024

@author: Brendan
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, roc_curve, roc_auc_score
from joblib import load
from itertools import product

def Prediction_FXN(Label_Predict):
    New_Label_Predict = []
    for i in range(Label_Predict.shape[0]):
        if Label_Predict[i][0] > Label_Predict[i][1] and Label_Predict[i][0] > Label_Predict[i][2] and Label_Predict[i][0] > Label_Predict[i][3]:
            New_Label_Predict.append("Bus")
        elif Label_Predict[i][1] > Label_Predict[i][0] and Label_Predict[i][1] > Label_Predict[i][2] and Label_Predict[i][1] > Label_Predict[i][3]:
            New_Label_Predict.append("Car")
        elif Label_Predict[i][2] > Label_Predict[i][0] and Label_Predict[i][2] > Label_Predict[i][1] and Label_Predict[i][2] > Label_Predict[i][3]:
            New_Label_Predict.append("Motorcycles")
        elif Label_Predict[i][3] > Label_Predict[i][0] and Label_Predict[i][3] > Label_Predict[i][1] and Label_Predict[i][3] > Label_Predict[i][2]:
            New_Label_Predict.append("Person")
            
    return New_Label_Predict

def Label_FXN(Label):
    New_Label = []
    for i in range(Label.shape[0]):
        if Label[i] == 0:
            New_Label.append("Bus")
        elif Label[i] == 1:
            New_Label.append("Car")
        elif Label[i] == 2:
            New_Label.append("Motorcycles")
        elif Label[i] == 3:
            New_Label.append("Person")

    return New_Label
def plot_cm(cm, data, save_dir = None):
    
    plt.imshow(cm, cmap = plt.cm.Blues, interpolation = 'nearest')
    plt.title(f"{data} Confusion Matrix")
    plt.colorbar()
    classes = ["Bus", "Car", "Motorcycles", "Person"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    for row, column in product(range(0, cm.shape[0]), range(0, cm.shape[1])):
        plt.text(row, column, cm[row][column], horizontalalignment = "center")
        
    plt.ylabel("True Label")
    plt.xlabel("Predict Label")
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/{data}_cm.png', format='png')
    plt.show()

def result(multi_cm, data, Class):

    TN, FP, FN, TP = multi_cm.ravel()
    AUC_Score = float(TP + TN) / float(TP + TN + FP + FN)
    Sensitivity = float(TP / float(TP + FN))
    Specificity = float(TN / float(TN + FP))
    PPV = float(TP / float(TP + FP))
    NPV = float(TN / float(TN + FN))
    
    print("----------------------------------------------")
    print(f"{data} Result - {Class}")
    print(f"TN = {TN}, FP = {FP}, FN = {FN}, TP = {TP}")
    print(f'AUC_Score = {AUC_Score*100:.2f}')
    print(f'Sensitivity = {Sensitivity*100:.2f}')
    print(f'Specificity = {Specificity*100:.2f}')
    print(f'PPV = {PPV*100:.2f}')
    print(f'NPV = {NPV*100:.2f}')
    print("----------------------------------------------")

def ROC(Label, Label_Predict, name, classes, multi_cm, save_dir = None):
    New_Label = np.zeros((Label.shape[0], 4), dtype = int)
    
    for indx, predict in enumerate(Label):
        if predict == 0:
            New_Label[indx][0] = 1
        elif predict == 1:
            New_Label[indx][1] = 1
        elif predict == 2:
            New_Label[indx][2] = 1
        else:
            New_Label[indx][3] = 1
    
    TN, FP, FN, TP = multi_cm[0].ravel()
    AUC_Score_Bus = float(TP + TN) / float(TP + TN + FP + FN)
    
    TN, FP, FN, TP = multi_cm[1].ravel()
    AUC_Score_Car = float(TP + TN) / float(TP + TN + FP + FN)
    
    TN, FP, FN, TP = multi_cm[2].ravel()
    AUC_Score_Motorcycles = float(TP + TN) / float(TP + TN + FP + FN)
    
    TN, FP, FN, TP = multi_cm[3].ravel()
    AUC_Score_Person = float(TP + TN) / float(TP + TN + FP + FN)
    
    FPR = dict()
    TPR = dict()
    AUC = dict()
    
    for Class in range(len(classes)):
        FPR[Class], TPR[Class], _ = roc_curve(New_Label[:, Class], Label_Predict[:, Class])
        AUC[Class] = roc_auc_score(New_Label[:, Class], Label_Predict[:, Class])

    plt.figure()
    plt.plot(FPR[0], TPR[0], color = "blue", label = f"{name} ROC Curve for {classes[0]} (AUC = {AUC_Score_Bus:.2f})")
    plt.plot(FPR[1], TPR[1], color = "red", label = f"{name} ROC Curve for {classes[1]} (AUC = {AUC_Score_Car:.2f})")
    plt.plot(FPR[2], TPR[2], color = "green", label = f"{name} ROC Curve for {classes[2]} (AUC = {AUC_Score_Motorcycles:.2f})")
    plt.plot(FPR[3], TPR[3], color = "yellow", label = f"{name} ROC Curve for {classes[3]} (AUC = {AUC_Score_Person:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    if save_dir:
        plt.savefig(f'{save_dir}/{name}_roc.png', format='png')
    plt.show()


def main():
    model = "test_model2"
    study = 'Dataset'
    classes = ["Bus", "Car", "Motorcycles", "Person"]
    

    Training_Label_Predict = load(f"E:\Brendan_2023_24\CNN\{model}\Training_Label_Predict.sav")
    Label_train = load(f"E:\Brendan_2023_24\{study}\Label_train.sav")
    
    
    Training_Label_Predict_FXN = Prediction_FXN(Training_Label_Predict)
    Label_train_FNX = Label_FXN(Label_train)
    tr_cm = confusion_matrix(Label_train_FNX, Training_Label_Predict_FXN, labels = classes)
    plot_cm(tr_cm, data = "Training")
    tr_multi_cm = multilabel_confusion_matrix(Label_train_FNX, Training_Label_Predict_FXN, labels = classes)
    ROC(Label_train, Training_Label_Predict, "Training", classes, tr_multi_cm)
    result(tr_multi_cm[0], "Training", classes[0])
    result(tr_multi_cm[1], "Training", classes[1])
    result(tr_multi_cm[2], "Training", classes[2])
    result(tr.multi_cm[3], "Training", classes[3])
    
    Testing_Label_Predict = load(f"E:\Brendan_2023_24\CNN\{model}\Testing_Label_Predict.sav")
    Label_test = load(f"E:\Brendan_2023_24\{study}\Label_test.sav")
    
    
    Testing_Label_Predict_FXN = Prediction_FXN(Testing_Label_Predict)
    Label_test_FXN = Label_FXN(Label_test)
    te_cm = confusion_matrix(Label_test_FXN, Testing_Label_Predict_FXN, labels = classes)
    plot_cm(te_cm, data = "Testing")
    te_multi_cm = multilabel_confusion_matrix(Label_test_FXN, Testing_Label_Predict_FXN, labels = classes)
    ROC(Label_test, Testing_Label_Predict, "Testing", classes, te_multi_cm)
    result(te_multi_cm[0], "Testing", classes[0])
    result(te_multi_cm[1], "Testing", classes[1])
    result(te_multi_cm[2], "Testing", classes[2])
    result(te_multi_cm[3], "Testing", classes[3])
    

if __name__ == "__main__":
    main()