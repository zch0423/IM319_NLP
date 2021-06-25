#coding=utf-8
'''
    FileName      ：decomPCA.py
    Author        ：@zch0423
    Date          ：Jun 11, 2021
    Description   ：
    PCA 查看bert embedding主成分
'''
#%%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#%%
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
#%%
w2v_type="avg"
# trn_X, trn_y = loadData("trn", w2v_type)
dev_X, dev_y = loadData("dev", w2v_type)
# tst_X, tst_y = loadData("tst", w2v_type)
x = np.arange(1, 400, 20)
y = []
for i in x:
    pca = PCA(n_components=i)
    pca.fit(dev_X)
    y.append(sum(pca.explained_variance_ratio_))
plt.plot(x, y)
plt.scatter(x,y)
for i,j in list(zip(x, y))[::4]:
    plt.text(i, j-0.05, "(%d,%.2f)" % (i,j))
plt.title("PCA For BERT Embedding")
plt.ylabel("Explained Ratio")
plt.xlabel("Components")
plt.show()
