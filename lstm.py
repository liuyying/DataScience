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


def scale(train, test,cat_columns):
    train_values: object = train.values
    test_values = test.values
    # fit scaler
    train_con = train_values[:, :9]
    test_con = test_values[:, :9]
    con_scaler = MinMaxScaler(feature_range= (0,1))
    con_scaled = con_scaler.fit(train_con)

    # transform 连续的train
    train_con = train_con.reshape(train_con.shape[0], train_con.shape[1])
    train_con_scaled = con_scaled.transform(train_con)
    # transform 连续的test
    test_con = test_con.reshape(test_con.shape[0], test_con.shape[1])
    test_con_scaled = con_scaled.transform(test_con)

    # transform 离散的train
    train_cat_scaled = cat_data(train,cat_columns)
    # transform 离散的train
    test_cat_scaled = cat_data(test,cat_columns)

    #合并
    train_scaled = concatenate((train_con_scaled, train_cat_scaled), axis=1)
    test_scaled = concatenate((test_con_scaled, test_cat_scaled), axis=1)

    return con_scaled, train_scaled, test_scaled

def cat_data(df,c):
    v=df.values[:,-4:]
    le = LabelEncoder()
    for col in c:
        df[col] = le.fit_transform(df[col].values)  # 对每一列所对应的离散型特征进行编码
    dis = df[c].values.tolist()
    cat_scaler = OneHotEncoder(categories='auto')  # 创建OneHotEncoder类的实例
    cat_scaled = cat_scaler.fit_transform(dis).toarray()
    cat_scaled = concatenate((cat_scaled, v), axis=1)
    return cat_scaled

    #
    # # split into train and test sets
    # values = purchase.values
    #
    # # split into input and outputs
    # train_X, train_y = values[:, 1:], values[:, 1]
    #
    #
    # # 将输入（X）重构为LSTM预期的3D格式，即[样本，时间步长，特征]。
    # # reshape input to be 3D [samples, timesteps, features]
    # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # return train_X, train_y, test_X, test_y,con_scaler


# load data set
# def load_dataset(file_path='dataset.csv', header_row_index=0, index_col_name =None, col_to_predict, cols_to_drop=None):
def _load_dataset(file_path, header_row_index, index_col_name, col_to_predict, cols_to_drop):
    """
    file_path: the csv file path
    header_row_index: the header row index in the csv file
    index_col_name: the index column (can be None if no index is there)
    col_to_predict: the column name/index to predict
    cols_to_drop: the column names/indices to drop (single label or list-like)
    """
    # read dataset from disk
    dataset = pd.read_csv(file_path, header=header_row_index, index_col=False)
    # print(dataset)

    # set index col，设置索引列，参数输入列的名字列表
    if index_col_name:
        dataset.set_index(index_col_name, inplace=True)

    # drop nonused colums，删除不需要的列，参数输入列的名字列表
    '''if cols_to_drop:
        if type(cols_to_drop[0]) == int:
            dataset.drop(index=cols_to_drop, axis=0, inplace=True)
        else:
            dataset.drop(columns=cols_to_drop, axis=1, inplace=True)'''
    if cols_to_drop:
        dataset.drop(cols_to_drop, axis=1, inplace=True)

    # print('\nprint data set again\n',dataset)
    # get rows and column names
    col_names = dataset.columns.values.tolist()
    values = dataset.values
    # print(col_names, '\n values\n', values)

    # move the column to predict to be the first col: 把预测列调至第一列
    col_to_predict_index = col_to_predict if type(col_to_predict) == int else col_names.index(col_to_predict)
    output_col_name = col_names[col_to_predict_index]
    if col_to_predict_index > 0:
        col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[
                                                                                           col_to_predict_index + 1:]
    values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)),
                             values[:, :col_to_predict_index], values[:, col_to_predict_index + 1:]), axis=1)
    # print(col_names, '\n values2\n', values)
    # ensure all data is float
    values = values.astype("float32")
    # print(col_names, '\n values3\n', values)
    return col_names, values, values.shape[1], output_col_name


# split into input and outputs
def split_data(train,test, verbose):
    # split into input and outputs
    train_X, train_y = train[:,1:], train[:,1]
    test_X, test_y = test[:, 1:], test[:,1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    if verbose:
        print("")
        print("train_X shape:", train_X.shape)
        print("train_y shape:", train_y.shape)
        print("test_X shape:", test_X.shape)
        print("test_y shape:", test_y.shape)

    return train_X, train_y, test_X, test_y


# create the nn model
# def _create_model(train_X, train_y, test_X, test_y, n_neurons=20, n_batch=50, n_epochs=60, is_stateful=False, has_memory_stack=False, loss_function='mae', optimizer_function='adam', draw_loss_plot=True, output_col_name, verbose=True):
def create_model(train_X, train_y, test_X, test_y,n_neurons, n_batch, n_epochs,loss_function,
                 optimizer_function, draw_loss_plot, output_col_name, verbose):
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
    model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2]),
                       return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(n_neurons))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
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
def make_prediction(model,  test_x, test_y, n_batch, con_scaler,
                     draw_prediction_fit_plot, output_col_name, verbose):
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

    yhat = model.predict(test_X, batch_size=n_batch, verbose=1 if verbose else 0)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))

    inv_yhat = concatenate((yhat, test_x[:,1:9]), axis=1)
    inv_yhat = con_scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # predict_y = np.array(predict_y)
    # 真实数据逆缩放 invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, 1:9]), axis=1)
    inv_y = con_scaler.inverse_transform(inv_y)
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
    train_data = pd.read_csv("data/train56/purchase_train.csv", header=0, index_col=0)
    test_data1 = pd.read_csv("data/train56/purchase_test.csv", header=0, index_col=0)   #最终测试集
    test_data = pd.read_csv("data/train56/purchase_validation.csv", header=0, index_col=0)    #验证集
    train_values = train_data.values
    cat_columns = train_data.columns[9:14].tolist()
    print(cat_columns)
    test_values = test_data.values
    print('values before series_to_supervised value shape:', train_data.shape)

    col_names = train_data.columns.values.tolist()

    scaler, train , test = scale(train_data,test_data,cat_columns)
    print("train归一化后", pd.DataFrame(train).shape)
    print("test归一化后", pd.DataFrame(test).shape)

    train_X, train_Y, test_X, test_Y = split_data(train,test,True)

    # !input
    n_neurons = 60
    n_batch = 60
    n_epochs = 100
    loss_function = 'mae'
    optimizer_function = 'adam'
    draw_loss_plot = True
    output_col_name = col_names [0]

    model = create_model(train_X, train_Y, test_X,test_Y, n_neurons, n_batch, n_epochs,
                         loss_function, optimizer_function,draw_loss_plot, output_col_name, True)


    model.save('./my_model_01')

    # !input
    draw_prediction_fit_plot = True
    actual_target, predicted_target, error_value, error_percentage = make_prediction(model, test_X, test_Y,
                                                                                      n_batch,scaler,
                                                                                      draw_prediction_fit_plot,
                                                                                      output_col_name,True)

