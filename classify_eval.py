import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

def classify_eval(X_test, y_test, fitted_model):
    
    pred = fitted_model.predict(X_test)
    pred_proba = fitted_model.predict_proba(X_test)
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred_proba[:,1])
    
    print('오차행렬')
    print(confusion)
    print('정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1스코어: {:.4f}'.format(f1))
    print('ROC AUC 값: {:.4f}'.format(roc_score))
    
    pred_proba_class1 = pred_proba[:,1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_class1)
    #print('반환된 분류 결정 임곗값 배열의 shape: ', thresholds.shape)
    
    # 임곗값 thresholds.shape[0]/10스탭 추출
    thr_index = np.arange(0, thresholds.shape[0], 10)
    #print('샘플 추출을 위한 임계값 배열의 index 25개: ', thr_index)
    #print('샘플용 25개의 임곗값": ', np.round(thresholds[thr_index], 2))
    
    # thresholds.shape[0]/10 스탭 단위로 추출된 임계값에 따른 정밀도와 재현율 값
    #print('샘플 임계값별 정밀도: ', np.round(precisions[thr_index], 3))
    #print('샘플 임계값별 재현율: ', np.round(recalls[thr_index], 3))
    
    pre_recal = pd.DataFrame()
    pre_recal['임계값'] = np.round(thresholds[thr_index], 2)
    pre_recal['정밀도'] = np.round(precisions[thr_index], 3)
    pre_recal['재현율'] = np.round(recalls[thr_index], 3)
    pre_recal
    
    print('\n --- 정확도 / 정밀도 트레이드 오프 그래프 ---')
    # x축을 threshold값으로, y축은 정밀도, 재현율 값으로 각각 plot 수행, 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')
    
    # threshold 값 x축의 scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()
    return pre_recal.T
