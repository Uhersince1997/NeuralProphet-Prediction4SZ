import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import datetime
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ts.set_token('7345b9fa7eba054aeacb922c113a3625f91c6f5793356b20e9bb4eb6')
start_date, end_date ='20100101', '20190408'
pro = ts.pro_api('7345b9fa7eba054aeacb922c113a3625f91c6f5793356b20e9bb4eb6')
data = pro.daily(ts_code='000001.SZ', start_date=start_date, end_date=end_date)
# data = data[["trade_date", "open", "high", "low", "close", "vol"]]
data = data[["trade_date", "close"]]
data = data.rename(columns={"trade_date": "ds", "close":"y"})
data = data.iloc[::-1]

time_array1 = time.strptime(start_date, "%Y%m%d")
timestamp_day1 = int(time.mktime(time_array1))
time_array2 = time.strptime(end_date, "%Y%m%d")
timestamp_day2 = int(time.mktime(time_array2))
fulldays = (timestamp_day2 - timestamp_day1) // 60 // 60 // 24
# print(data.head(10))
# 把datetime转成字符串
def datetime_toString(dt):
    return dt.strftime("%Y%m%d")

# 把字符串转成datetime
def string_toDatetime(string):
    return datetime.datetime.strptime(string, "%Y%m%d")


# 缺失值处理，插值替换
date_start = data.iloc[0, 0]  # 初始时间
data_date = data['ds'].tolist()  # 数据日期转为列表
data_data = data['y'].tolist()  # 数据值转为列表
for j in range(0, len(data_date) - 1):
    if len(data_date) < fulldays:
        dates = data_date[j]
        datee = data_date[j + 1]

        day1 = datetime.timedelta(days=1)
        dates = string_toDatetime(dates)
        dates = dates + datetime.timedelta(days=1)  # 日期加一
        dates = datetime_toString(dates)
        while dates != datee:
            nada = data_data[j - 1]
            adda = [dates, nada]
            date_da = pd.DataFrame(adda).T
            date_da.columns = data.columns
            data = pd.concat([data, date_da])  # 将缺失日期加入数据列表中
            dates = string_toDatetime(dates)
            dates += day1  # 日期加一
            dates = datetime_toString(dates)  # 日期字符串转日期时间类型
data = data.sort_values(by=['ds'])

# print(data.head(20))
NPmodel = NeuralProphet(
    n_forecasts=7,
    n_lags=30,
    n_changepoints=50,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    batch_size=64,
    epochs=100,
    learning_rate=1.0,
)
metrics = NPmodel.fit(data, freq="D")
future = NPmodel.make_future_dataframe(data, periods=1, n_historic_predictions=len(data))
prediction = NPmodel.predict(future)
# Plotting
forecast = NPmodel.plot(prediction)
plt.title("Prediction of the SZ000001 Stock Price using NeuralProphet for the next 60 days")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()

