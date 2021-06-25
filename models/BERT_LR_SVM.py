#coding=utf-8
'''
    FileName      ：bert_LR_SVM.py
    Author        ：@zch0423
    Date          ：Jun 25, 2021
    Description   ：
'''
# import pandas as pd
import pickle
import numpy as np
from numpy.core.defchararray import mod
from scipy.sparse import data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score


def loadData(data_type="trn", w2v_type="avg"):
    '''
    @Description
    载入数据并返回
    ------------
    @Params
    data_type, str, in ["trn", "dev", "tst"]
    w2v_type, str, in ["avg", "CLS"]
    ------------ 
    @Returns
    X, y
    '''
    if data_type not in ["trn", "dev", "tst"]:
        raise ValueError("check data_type!")
    if w2v_type not in ["avg", "CLS"]:
        raise ValueError("check w2v_type")
    base = "/Users/zch/Desktop/IM319_NLP.nosync/project/data"
    X_path = f"{base}/w2v/{w2v_type}_{data_type}_last.npy"
    y_path = f"{base}/labels/{data_type}.npy"
    X = np.load(X_path)
    X = X.reshape(X.shape[0], X.shape[-1])
    y = np.load(y_path)
    return X, y

def label2weight(y):
    '''
    @Description
    将label转为成0-1加weights
    ------------
    @Params
    y, array
    ------------
    @Returns
    y1, weights
    '''
    weights = []
    y1 = []
    for w in y:
        if np.isnan(w):
            weights.append(0)
            y1.append(0)
        elif w == 0:
            weights.append(1)
            y1.append(0)
        else:
            weights.append(w)
            y1.append(1)
    return np.array(y1), np.array(weights)

    
def fitOne(trn_X, trn_y, dev_X, dev_y, y_idx=2, m_type="lr"):
    '''
    @Description
    LR or SVM fit
    ------------
    @Params
    trn_X, trn_y, 训练集
    dev_X, dev_y, 开发集
    y_idx, int 0-3, 不同类别 whoTarget intentYN sexYN offensiveYN
    m_type, str, model type lr or svm
    ------------
    @Returns
    model
    '''
    trn_y1, trn_w = label2weight(trn_y[y_idx])
    dev_y1, dev_w = label2weight(dev_y[y_idx])
    if m_type == "lr":
        model = LogisticRegression(max_iter=10000, verbose=False, n_jobs=-1)
    else:
        model = SVC(verbose=False, probability=True)
    model.fit(trn_X, trn_y1, sample_weight=trn_w)
    print("%.4f"%model.score(trn_X, trn_y1, sample_weight=trn_w))
    print("%.4f"%model.score(dev_X, dev_y1, sample_weight=dev_w))
    return model 


if __name__ == "__main__":
    # w2v_type = sys.argv[1]  # w2v 类型
    # m_type = sys.argv[2]  # SVM或LR
    # y_idx = int(sys.argv[3])  # 标签
    # if w2v_type not in ["avg", "CLS"]:
    #     raise ValueError("w2v")
    # if m_type not in ["lr", "svm"]:
    #     raise ValueError("model")
    # if y_idx<0 or y_idx >3:
    #     raise ValueError("y_idx")
    outPath = "/Users/zch/Desktop/IM319_NLP.nosync/project/models/"
    labels = ["whoTarget", "intentYN", "sexYN", "offensiveYN"]
    # for m_type in ["lr", "svm"]:
    for m_type in ["svm"]:
        for w2v_type in ["avg", "CLS"]:
            trn_X, trn_y = loadData("trn", w2v_type)
            dev_X, dev_y = loadData("dev", w2v_type)
            tst_X, tst_y = loadData("tst", w2v_type)
            for y_idx in range(4):
                model_name = f"{w2v_type}-{m_type.upper()}-{labels[y_idx]}"
                print("Running", model_name)
                model = fitOne(trn_X, trn_y, dev_X, dev_y, y_idx=y_idx, m_type=m_type)
                tst_y1, tst_w = label2weight(tst_y[y_idx])
                print("%.4f"%model.score(tst_X, tst_y1, sample_weight=tst_w))
                with open(outPath+model_name+".bin", "wb") as f:
                    pickle.dump(model, f)
