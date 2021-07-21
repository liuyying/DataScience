import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import concatenate


def scale(values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaler, scaled


def cat_data(df,c):
    v=df.values[:,-3:]
    le = LabelEncoder()
    for col in c:
        df[col] = le.fit_transform(df[col].values)  # 对每一列所对应的离散型特征进行编码
    dis = df[c].values
    cat_scaler = OneHotEncoder(categories='auto')  # 创建OneHotEncoder类的实例
    cat_scaled = cat_scaler.fit_transform(dis).toarray()
    cat_scaled = concatenate((cat_scaled, v), axis=1)
    cat_names= ["x%s" % i for i in range(1,cat_scaled.shape[1]+1)]
    return cat_names,cat_scaled


def load_data(data,num_con):
    values = data.values
    con_names = data.columns[:num_con].tolist()
    print(data.shape)
    con_values = values[:, :num_con]  # 前包括，后不包括
    # 离散值所在列名
    cat_columns = data.columns[num_con:].tolist()
    cat_names,cat_values = cat_data(data, cat_columns)  # 离散数据热编码
    # print(cat_values.shape)
    # print(con_values.shape)
    # print(len(con_names),len(cat_names))
    con_data = concatenate((con_values, cat_values), axis=1)
    print("合并后", con_data.shape)
    scaler,normal_data = scale(con_data)
    normal_data=pd.DataFrame(normal_data,columns=con_names+cat_names)
    return scaler,normal_data


data = pd.read_csv("data/purchase_balance4.csv",header=0,index_col=0)
redeem_balance = pd.read_csv("data/redeem_balance2.csv")
# print(purchase_balance.shape)
# num_con=23
# scaler,data=load_data(purchase_balance,num_con)
# print(data)

data_fea1 = data.iloc[:, 2:]  # 取数据中指标所在的列

model = RandomForestRegressor(random_state=1, max_depth=10)
data_fea1 = data_fea1.fillna(0)  # 随机森林只接受数字输入，不接受空值、逻辑值、文字等类型
data_fea1 = pd.get_dummies(data_fea1)
model.fit(data_fea1, data["total_purchase"])

# 根据特征的重要性绘制柱状图
features = data_fea1.columns
importances = model.feature_importances_
indices = np.argsort(importances)
plt.title('Index selection')
plt.barh(range(len(indices)), importances[indices], color="dodgerblue", align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative importance of indicators')
plt.show()
