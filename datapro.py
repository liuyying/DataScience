import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

user_balance = pd.read_csv("data/user_balance_table.csv")

user_balance.drop(['yBalance','direct_purchase_amt','purchase_bal_amt','purchase_bank_amt',
                   'tftobal_amt','tftocard_amt','transfer_amt','share_amt','category1','category2','category3','category4'],axis=1,inplace=True)

#转换为日期格式
user_balance['report_date'] = user_balance['report_date'].apply(str)
user_balance['report_date']= pd.to_datetime(user_balance['report_date'])
#删除没有申购赎回操作的行
user_balance1=user_balance[~(user_balance['total_purchase_amt'].isin([0]) & user_balance['total_redeem_amt'].isin([0]))]

print(user_balance.shape)
print(user_balance1.shape)

# s1=user_balance1['total_purchase_amt'].sum()
# s2=user_balance['total_purchase_amt'].sum()
# print(s1)
# print(s2)
# #查看数据类型
print(user_balance1.info())

#求用户平均每周的操作次数
tcount = user_balance1.groupby(['user_id'])['total_redeem_amt'].count()
max_date = user_balance1.groupby(['user_id'])['report_date'].max()
min_date = user_balance1.groupby(['user_id'])['report_date'].min()
delta = max_date-min_date+dt.timedelta(1)
mean_count = tcount/delta.dt.days*7

#
# #用户申购总量
# sum_redeems =user_balance1.groupby(['user_id'])['total_redeem_amt'].sum()
# sum_purchases =user_balance1.groupby(['user_id'])['total_purchase_amt'].sum()
#
#求用户余额宝平均持有量
mean_balance =user_balance1.groupby(['user_id'])['tBalance'].mean()

users = pd.DataFrame({'mean_balance':mean_balance,'mean_count':mean_count})
users.reset_index(inplace=True)
print(users.info())
print(mean_balance.describe())
print(user_balance1['tBalance'].describe())

#大小额用户判断
def function(a):
	if a>=1000000:
		return 1
	else:
		return 0
users['is_rich'] = users.apply(lambda x: function(x.mean_balance), axis = 1)

#活跃用户判断
def function1(a):
    if a>=6:
        return 2
    elif a>=3:
        return 1
    else:
        return 0
users['is_activity']= users.apply(lambda x: function1(x.mean_count), axis = 1)
# users.to_csv("data/new/user1.csv")

users.plot(x='mean_balance',y='mean_count',kind='scatter')
plt.show()
# users.to_csv("data/new/users2.csv")

user_balance2 =  pd.merge(user_balance1,users, how='left',on='user_id')

user_balance2.fillna(0,inplace=True)
# user_balance2.to_csv("data/user_balance.csv")
print(users.shape)
print(user_balance2['total_purchase_amt'].sum())

#汇总每日的赎回消费总数
tconsume = user_balance2.groupby(['report_date'])['consume_amt'].sum()

#汇总每日申购总数
total_tpurchase = user_balance2.groupby(['report_date'])['total_purchase_amt'].sum()

#汇总每日赎回总额
total_tredeem = user_balance2.groupby(['report_date'])['total_redeem_amt'].sum()

#汇总每日用户数
count_tusers = user_balance2.groupby(['report_date'])['user_id'].count()

#汇总每天的富人人数
count_triches = user_balance2.groupby(['report_date'])['is_rich'].sum()

#汇总富人每天的申购赎回总额
user_balance3=user_balance2[(user_balance2['is_rich'].isin([1]))]
rich_tpurchase = user_balance3.groupby(['report_date'])['total_purchase_amt'].sum()
rich_tredeem = user_balance3.groupby(['report_date'])['total_redeem_amt'].sum()

#汇总每天的活跃人数
user_balance5=user_balance2[(user_balance2['is_activity'].isin([2]))]
count_tactive = user_balance5.groupby(['report_date'])['is_activity'].count()
#汇总活跃用户每日的申购与赎回总额
active_tpurchase = user_balance5.groupby(['report_date'])['total_purchase_amt'].sum()
active_tredeem = user_balance5.groupby(['report_date'])['total_redeem_amt'].sum()

#汇总每天的普通人数
user_balance4=user_balance2[(user_balance2['is_activity'].isin([1]))]
count_tgeneral= user_balance4.groupby(['report_date'])['is_activity'].count()
general_tpurchase = user_balance4.groupby(['report_date'])['total_purchase_amt'].sum()
general_tredeem = user_balance4.groupby(['report_date'])['total_redeem_amt'].sum()


sum_balance = pd.DataFrame({'total_tpurchase':total_tpurchase,'total_tredeem':total_tredeem,
                            'count_tusers':count_tusers,'count_triches':count_triches,
                            'count_tactive':count_tactive,'count_tgeneral':count_tgeneral,
                            'rich_tpurchase':rich_tpurchase,'rich_tredeem':rich_tredeem,
                            'active_tpurchase':active_tpurchase,'active_tredeem':active_tredeem,
                            'general_tpurchase':general_tpurchase,'general_tredeem':general_tredeem,
                            'tconsume':tconsume})


sum_balance.reset_index(inplace=True)
sum_balance.fillna(0,inplace=True)
print(sum_balance.shape)

# describe=sum_balance.describe()
# describe.to_csv("data/sum_balance_describe.csv")

# #写入文件
sum_balance.to_csv("data/new/sum_balance2.csv")
print("success")


# #将银行间拆借利率合并进来
# sum_balance = pd.read_csv("data/sum_balance_table1.csv")
# sum_balance ['report_date'] = sum_balance ['report_date'].apply(str)
# sum_balance ['report_date']= pd.to_datetime(sum_balance ['report_date'])
# shibor = pd.read_csv("data/mfd_bank_shibor.csv")
# shibor['report_date'] = shibor['report_date'].apply(str)
# shibor['report_date']= pd.to_datetime(shibor['report_date'])
# print(shibor.info())
# print(shibor['report_date'])
# sum_balance1 = pd.merge(sum_balance,shibor, how='left',on='report_date')
# print(sum_balance1[-10:])
# print(sum_balance1.ix[426,40])
# for index in range(0,427):
#     for i in range(33,41):
#         if pd.isnull(sum_balance1.ix[index,i]):
#             sum_balance1.ix[index,i]=sum_balance1.ix[index-1,i]
# print(sum_balance1[-10:])
# sum_balance1.to_csv("data/sum_balance.csv")
#
# purchase_balance = pd.read_csv("data/purchase_balance.csv")
# pdescribe = purchase_balance.describe()
# pdescribe.to_csv("data/purchase_balance_describe.csv")
# redeem_balance = pd.read_csv("data/redeem_balance.csv")
# rdescribe = redeem_balance.describe()
# rdescribe.to_csv("data/redeem_balance_describe.csv")
