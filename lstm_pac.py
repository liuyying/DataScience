from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation,Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import concatenate
from math import sqrt, ceil
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit(train)

    # transform 连续的train
    train= train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaled.transform(train)
    # transform 连续的test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaled.transform(test)

    return scaled, train_scaled, test_scaled


def cat_data(df,c):
    v=df.values[:,-3:]
    le = LabelEncoder()
    for col in c:
        df[col] = le.fit_transform(df[col].values)  # 对每一列所对应的离散型特征进行编码
    dis = df[c].values.tolist()
    cat_scaler = OneHotEncoder(categories='auto')  # 创建OneHotEncoder类的实例
    cat_scaled = cat_scaler.fit_transform(dis).toarray()
    cat_scaled = concatenate((cat_scaled, v), axis=1)
    return cat_scaled

# convert series to supervised learning (ex: var1(t)_row1 = var1(t-1)_row2)，列表打印出来一看就明白了
#def _series_to_supervised(values, n_in=3, n_out=1, dropnan=True, col_names, verbose=True):
def series_to_supervised(values, n_in, n_out,dropnan, col_names,verbose):
    """
    values: dataset scaled values
    n_in:  与多少个之前的time_step相关
    n_out: 预测未来多少个time_step
    dropnan: whether to drop rows with NaN values after conversion to supervised learning
    col_names: name of columns for dataset
    verbose: whether to output some debug data
    num_lag:滞后变量数量
    """

    n_vars = 1 if type(values) is list else values.shape[1]
    if col_names is None: col_names = ["var%d" % (j + 1) for j in range(n_vars)]
    df = pd.DataFrame(values)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))  # 这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols,
                 axis=1)  # 将cols中的每一行元素一字排开，连接起来，vala t-n_in, valb t-n_in ... valta t, valb t... vala t+n_out-1, valb t+n_out-1
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def split_to_train_test(num_test,values,n_features):
    drop=n_features-1
    if drop>0:
        train = values[:-num_test, :-drop]
        test = values[-num_test:, :-drop]
    else:
        train = values[:-num_test, :]
        test = values[-num_test:, :]

    test = pd.DataFrame(test)
    train = pd.DataFrame(train)
    print("\nsupervised train_data shape:",train.shape)
    print("\nsupervised test_data shape:,",test.shape)

    return test,train


def split_to_XY(train,test,timesteps,n_features,verbose):

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timesteps, n_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, n_features))

    if verbose:
        print("")
        print("train_X shape:", train_X.shape)
        print("train_y shape:", train_y.shape)
        print("test_X shape:", test_X.shape)
        print("test_y shape:", test_y.shape)

    return train_X, train_y, test_X, test_y


def create_model(train_X, train_y, test_X, test_y,n_neurons1,n_neurons2, n_batch, n_epochs,loss_function,
                 optimizer_function, draw_loss_plot, output_col_name,n_out, verbose):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    n_neurons: number of neurons for LSTM nn
    n_batch: 每批次的数据量大小
    n_epochs: 训练批次
    has_memory_stack: 模型是否有记忆堆栈
    loss_function: 模型的损失函数
    optimizer_function: 模型的损失优化器函数
    draw_loss_plot: 是否绘制损失历史绘图
    output_col_name: 要预测的输出目标列名称
    verbose: 是否输出一些调试数据
    #根据经验选择，隐藏节点数大概是输入节点数的两倍左右
    #层数1,2,3,4 基本没有看到过更多的了，大部分网络可能就是两层双向或者一层双向即可
    #结点数 根据特征维数和数据量 64,128,256,512, 也是基本没有看到过更多的，大部分网络就是128或者256
    """

    # design network
    model = Sequential()
    model.add(LSTM(n_neurons1, input_shape=(train_X.shape[1], train_X.shape[2]),
                       return_sequences=True))
    # model.add(LSTM(n_neurons1, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(n_neurons2))
    model.add(Dropout(0.5))
    model.add(Dense(n_out))
    model.compile(loss=loss_function, optimizer=optimizer_function)
    if verbose:
        print("")

    history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch,validation_data=(test_X, test_y),
                        verbose=2 if verbose else 0, shuffle=False)

    if draw_loss_plot:
        plt.plot(history.history["loss"] , label="Train Loss (%s)" % output_col_name)
        plt.plot(history.history["val_loss"] ,label="Test Loss (%s)" % output_col_name)
        plt.legend()
        plt.show()

    print(history.history)
    # model.save('./my_model_%s.h5'%datetime.datetime.now())
    return model

# make a prediction
# def _make_prediction(model, test_X, test_y, compatible_n_batch, n_intervals=3, n_features, scaler=(0,1), draw_prediction_fit_plot=True, output_col_name, verbose=True):
def make_prediction(model,  test_x, test_y, n_batch, scaler,n_intervals
                     ,draw_prediction_fit_plot, output_col_name
                    ,n_features,num_test,con_data1,verbose):
    """
    test_X: test inputs
    test_y: test targets
    n_batch: 修改(兼容)nn批处理大小
    scaler: 用于将转换转换为真实比例的scaler对象
    draw_prediction_fit_plot: 是否绘制预测拟合曲线与实际拟合曲线
    output_col_name: 要预测的输出/目标列的名称
    verbose: 是否输出一些调试数据
    """

    if verbose:
        print("")

    yhat = model.predict(test_x, batch_size=n_batch, verbose=1 if verbose else 0)
    test_x = test_x.reshape((test_x.shape[0], n_intervals * n_features))
    #预测值
    inv_yhat = concatenate((yhat,stand_data[-num_test:,1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # predict_y = np.array(predict_y)
    # 真实数据逆缩放 invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, stand_data[-num_test:,1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]


    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    mse=mean_squared_error(inv_y, inv_yhat)
    mae=mean_absolute_error(inv_y, inv_yhat)
    r2=r2_score(inv_y, inv_yhat)

    # calculate average error percentage
    avg = np.average(inv_y)
    error_percentage = rmse / avg

    if verbose:
        print("")
        print("rmse: %.3f" % rmse)
        print("mse: %.3f" % mse)
        print("mae: %.3f" % mae)
        print("r2：%.3f" % r2)
        print("Test Average Value for %s: %.3f" % (output_col_name, avg))
        print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))

    if draw_prediction_fit_plot:
        plt.plot(inv_y, label="Actual (%s)" % output_col_name)
        plt.plot(inv_yhat, label="Predicted (%s)" % output_col_name)
        plt.legend()
        plt.show()

    return (inv_y, inv_yhat, rmse, error_percentage)



if __name__ == '__main__':
    # !input
    # data = pd.read_csv("data/purchase_balance7.csv", header=0, index_col=0)
    data = pd.read_csv("data/redeem_balance4.csv", header=0, index_col=0)
    values = data.values
    col_names=data.columns
    print(data.shape)
    num_con=15  #连续特征个数
    con_values = values[:,:num_con]  #前包括，后不包括

    #离散值所在列名
    cat_columns=data.columns[num_con:].tolist()
    cat_values = cat_data(data,cat_columns)   #离散数据热编码
    print("cat",cat_values.shape)

    # cat=data.iloc[n_in:, 2:]
    con_data = concatenate((con_values,cat_values), axis=1)
    print("合并后", con_data.shape)

    #标准化
    scaler=StandardScaler()
    stand_data=scaler.fit_transform(con_data)   #对onehot后的数据归一化
    pca_scaler=PCA(n_components=5)
    pca_data=pca_scaler.fit_transform(stand_data[:,1:num_con])
    print(pca_scaler.explained_variance_ratio_)
    print(pca_data)
    print("pca后",pca_data.shape)

    plt.bar(range(5),pca_scaler.explained_variance_ratio_, fc="dodgerblue", label='Single interpretation variance',
            align='center')
    plt.plot(range(5),np.cumsum(pca_scaler.explained_variance_ratio_), color="dodgerblue",
             label='Cumulative Explained Variance')
    # plt.title("total_purchase")
    plt.title("total_redeem")
    plt.xlabel('component')
    plt.ylabel('explained variance')
    plt.legend()
    plt.show()

    components = pca_scaler.components_

    print("前{}个主成分解释了数据中{:.2f}%的变化".format(5, sum(pca_scaler.explained_variance_ratio_) * 100))


    print(stand_data[:,0].shape)
    con_data1=concatenate((stand_data[:,0].reshape(stand_data.shape[0],1),pca_data),axis=1)
    con_data1 = concatenate((con_data1, con_data[:,15:]), axis=1)
    print("con_data1",con_data1.shape)
    n_features = con_data1.shape[1]

    #装换为监督学习型数据
    n_in=3
    n_out=1
    dropnan=True
    verbose=True
    supervised_data=series_to_supervised(con_data1, n_in, n_out, dropnan, None,verbose)
    print("转为监督学习数据后", supervised_data.shape)

    #划分训练集/验证集/测试集
    num_test=31
    test,train=split_to_train_test(num_test,supervised_data.values,n_features)

    train_X, train_Y, test_X, test_Y = split_to_XY(train.values, test.values,n_in,n_features,True)


    # !input
    n_neurons1 = 90
    n_neurons2 = 90
    n_batch = 20
    n_epochs =45
    loss_function = 'mse'  #mse
    optimizer_function = 'adam'
    draw_loss_plot = True
    output_col_name = col_names[0]

    model = create_model(train_X, train_Y, test_X, test_Y, n_neurons1,n_neurons2,n_batch, n_epochs,
                         loss_function, optimizer_function, draw_loss_plot, output_col_name, n_out,True)

    # model.save('./my_model_01')

    # !input
    draw_prediction_fit_plot = True
    actual_target, predicted_target, error_value, error_percentage = make_prediction(model, test_X, test_Y,
                                                                                     n_batch, scaler,n_in,
                                                                                     draw_prediction_fit_plot,
                                                                                     output_col_name,n_features,
                                                                                     num_test,stand_data,True)

