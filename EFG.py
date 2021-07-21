import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import concatenate
# For EFSG
from sklearn.linear_model import (LinearRegression, Ridge, LassoLarsIC, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestRegressor

def cat_data(df,c):
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].values)  # 对每一列所对应的离散型特征进行编码
    cat_scaler = OneHotEncoder(categories='auto')  # 创建OneHotEncoder类的实例
    df[c] = cat_scaler.fit_transform(df[c].values)

    return df

balance = pd.read_csv("data/2/purchase_balance.csv",header=0, index_col=0)
# balance = pd.read_csv("data/2/redeem_balance1.csv",header=0, index_col=0)
print("balance",balance.shape)


names = balance.columns[1:]
ranks={}
Y=balance.values[:,0]
X=balance.values[:,1:]
print("name:",len(names))
print("X",X.shape)
print("Y",Y.shape)

#
# X[:,10:] = X[:,:4] + np.random.normal(0, .025, (size,4))

# Defining the ranked dictionary, the coefficients are normalized.
def rank_to_dict(ranks, names, order=1):
  minmax = MinMaxScaler()
  ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
  ranks = map(lambda x: round(x, 2), ranks)
  return dict(zip(names, ranks ))
### Linear Modeling ###

### Simple Linear Regression ###
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["LR"] = rank_to_dict(np.abs(lr.coef_), names)
print("LR",ranks["LR"])

### Ridge Regression ###
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
print('Ridge',ranks["Ridge"])

### Lasso Regression based on AIC ###
lasso_aic = LassoLarsIC(criterion='aic',  max_iter=50000)
lasso_aic.fit(X, Y)
ranks["Lasso_aic"] = rank_to_dict(np.abs(lasso_aic.coef_), names)
print('Lasso_aic',ranks["Lasso_aic"])

### Lasso Regression based on BIC ###
lasso_bic = LassoLarsIC(criterion='bic', max_iter=50000)
lasso_bic.fit(X, Y)
ranks["Lasso_bic"] = rank_to_dict(np.abs(lasso_bic.coef_), names)
print("Lasso_bic",ranks["Lasso_bic"])

###### Distance Correlation Implementation ######距离相关系数
def dist(x, y):
    # 1d only
    return np.abs(x[:, None] - y)


def d_n(x):
    d = dist(x, x)
    dn = d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    return dn


def dcov_all(x, y):
    dnx = d_n(x)
    dny = d_n(y)

    denom = np.product(dnx.shape)
    dc = (dnx * dny).sum() / denom
    dvx = (dnx ** 2).sum() / denom
    dvy = (dny ** 2).sum() / denom
    dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
    return dc, dr, dvx, dvy
    # return dr

# stop the search when 5 features are left (they will get equal scores)

### Random Forest ###
rf = RandomForestRegressor(random_state=1, max_depth=10)
rf.fit(X, Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
print("RF",ranks["RF"])

### Calculate the Recursive Feature Elimination ###
print(X.shape)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X, Y)
ranks["RFE"] = rank_to_dict(rfe.ranking_.astype(float), names, order=-1)
print("RFE",ranks["RFE"])

### Correlation ###
f, pval = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)
print("Corr,",ranks["Corr."])

### Calculation the distance correlation ###

dis_corr = []

for i in range(0, len(names)):
    dc, dr, dvx, dvy = dcov_all(X[:, i], Y)
    dis_corr.append(dr)

ranks["Dis_Corr"] = rank_to_dict(dis_corr, names)
print("Dis_Corr",ranks["Dis_Corr"])

### Mean Calculation ###

r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print("\t%s" % "\t".join(methods))

for name in names:
    print("%s\t%s" % (name, "\t".join(map(str, [ranks[method][name] for method in methods]))))

rankspd = pd.DataFrame(ranks)
print("rankspd",rankspd)
# rankspd.to_csv("data/2/redeem_ranks2.csv")
rankspd.to_csv("data/2/purchase_ranks3.csv")