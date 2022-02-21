#Importing Libraries
import json
import gspread
import numpy as np
import tradermade as tm
from datetime import datetime, timedelta
from pytz import timezone
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import schedule
import time

n_steps=10
scaler=MinMaxScaler(feature_range=(0,1))
symbols = ['EURUSD', 'GBPJPY', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD', 'XAUUSD', 'XAGUSD', 'SPX500', 'NAS100']
# symbols = ['XAUUSD']
api_key = ""
tm.set_rest_api_key(api_key)

# Google Sheet:
sa = gspread.service_account(filename='service_account.json')
# sa = gspread.service_account()
sh = sa.open("FX - AI/ML Model Sheet")
wks = sh.worksheet("30M-1W")
print("Loading models...")

# Load all trained models once each week:
# models_1M = []
# print("Loading 1M models...")
# for index, pair in enumerate(symbols):
#     model = load_model('models/1M/'+ pair + '_1M_model.h5')
#     models_1M.append(model)

# models_15M = []
# print("Loading 15M models...")
# for index, pair in enumerate(symbols):
#     model = load_model('models/15M/'+ pair + '_15M_model.h5')
#     models_15M.append(model)

models_30M = []
print("Loading 30M models...")
for index, pair in enumerate(symbols):
    model = load_model('models/30M/'+ pair + '_30M_model.h5')
    models_30M.append(model)

models_1H = []
print("Loading 1H models...")
for index, pair in enumerate(symbols):
    model = load_model('models/1H/'+ pair + '_1H_model.h5')
    models_1H.append(model)

models_2H = []
print("Loading 2H models...")
for index, pair in enumerate(symbols):
    model = load_model('models/2H/'+ pair + '_2H_model.h5')
    models_2H.append(model)

models_4H = []
print("Loading 4H models...")
for index, pair in enumerate(symbols):
    model = load_model('models/4H/'+ pair + '_4H_model.h5')
    models_4H.append(model)
    
models_1D = []
print("Loading 1D models...")
for index, pair in enumerate(symbols):
    model = load_model('models/1D/'+ pair + '_1D_model.h5')
    models_1D.append(model)

models_1W = []
print("Loading 1W models...")
for index, pair in enumerate(symbols):
    model = load_model('models/1W/'+ pair + '_1W_model.h5')
    models_1W.append(model)

# Functions:
def predictionFunction(x_input, model):
    x_input = x_input.reshape((1, n_steps, 1))
    yhat = model.predict(x_input, verbose=0)
    return yhat

def transformValue(last_val, pred):
    pred = scaler.inverse_transform(pred)
    expected_move = pred[0] - last_val[0][0]
    trend = 'Bullish' if expected_move > 0 else 'Bearish'
    return trend, expected_move[0]

# Updating forecasting in google sheets:
def updateSheet(wks, cell_range, cell_values): 
    cell_list = wks.range(cell_range)
    for i, val in enumerate(cell_values):  #gives us a tuple of an index and value
        cell_list[i].value = val
    wks.update_cells(cell_list)

def botLogic(df, model):
    df=scaler.fit_transform(np.array(df).reshape(-1,1))
    lst = []
    lst.append(df)
    df = np.array(lst)
    pred = predictionFunction(df, model)
    last_val = scaler.inverse_transform(df[-1].reshape(1,-1))
    trend, expected_move = transformValue(last_val, pred)
    expected_move = json.dumps(np.round(expected_move.astype(float),5))
    return trend, expected_move

# def job1():
#     print("1 min...")
#     from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=1)).strftime("%Y-%m-%d-%H:%M")
#     to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
#     df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="minute",fields=["close"],period=1)
#     for index, pair in enumerate(symbols):
#         data = df[pair]
#         data = data[data.notna()]
#         if len(data) < 10:
#             print("Not enough data for " + pair)
#             continue
#         data = data[-10:]
#         botLogic(data, models_1M[index], 'C', 'D', str(index+10))

# def job2():
#     print("15 min...")
#     from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=1)).strftime("%Y-%m-%d-%H:%M")
#     to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
#     df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="minute",fields=["close"],period=15)
#     for index, pair in enumerate(symbols):
#         data = df[pair]
#         data = data[data.notna()]
#         if len(data) < 10:
#             print("Not enough data for " + pair)
#             continue
#         data = data[-10:]
#         botLogic(data, models_15M[index], 'E', 'F', str(index+10))

def job3():
    print("30 min...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=2)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="minute",fields=["close"],period=30)
    cell_list = []
    for index, pair in enumerate(symbols):
        data = df[pair]
        data = data[data.notna()]
        if len(data) < 10:
            print("Not enough data for " + pair)
            continue
        data = data[-10:]
        trend, value = botLogic(data, models_30M[index])
        cell_list.append(trend)
        cell_list.append(value)
    updateSheet(wks, 'C10:D21', cell_list)
    print("Updated in google sheet")

def job4():
    print("1 hour...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=4)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="hourly",fields=["close"],period=1)
    cell_list = []
    for index, pair in enumerate(symbols):
        data = df[pair]
        data = data[data.notna()]
        if len(data) < 10:
            print("Not enough data for " + pair)
            continue
        data = data[-10:]
        trend, value = botLogic(data, models_1H[index])
        cell_list.append(trend)
        cell_list.append(value)
    updateSheet(wks, 'E10:F21', cell_list)
    print("Updated in google sheet")

def job4a():
    print("2 hour...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=4)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="hourly",fields=["close"],period=1)
    cell_list = []
    for index, pair in enumerate(symbols):
        data = df[pair]
        data = data[data.notna()]
        data = data.iloc[::2]
        if len(data) < 10:
            print("Not enough data for " + pair)
            continue
        data = data[-10:]
        trend, value = botLogic(data, models_2H[index])
        cell_list.append(trend)
        cell_list.append(value)
    updateSheet(wks, 'G10:H21', cell_list)
    print("Updated in google sheet")

def job5():
    print("4 hour...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=5)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="hourly",fields=["close"],period=1)
    cell_list = []
    for index, pair in enumerate(symbols):
        data = df[pair]
        data = data[data.notna()]
        data = data.iloc[::4]
        if len(data) < 10:
            print("Not enough data for " + pair)
            continue
        data = data[-10:]
        trend, value = botLogic(data, models_4H[index])
        cell_list.append(trend)
        cell_list.append(value)
    updateSheet(wks, 'I10:J21', cell_list)
    print("Updated in google sheet")


def job6():
    print("1 day...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=20)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="daily",fields=["close"],period=1)
    cell_list = []
    for index, pair in enumerate(symbols):
        data = df[pair]
        data = data[data.notna()]
        if len(data) < 10:
            print("Not enough data for " + pair)
            continue
        data = data[-10:]
        trend, value = botLogic(data, models_1D[index])
        cell_list.append(trend)
        cell_list.append(value)
    updateSheet(wks, 'K10:L21', cell_list)
    print("Updated in google sheet")

def job7():
    print("1 week...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=140)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD,XAGUSD,SPX500,NAS100', start=from_date,end=to_datetime,interval="daily",fields=["close"],period=1)
    cell_list = []
    for index, pair in enumerate(symbols):
        data = df[pair]
        data = data[data.notna()]
        data = data.iloc[::5]
        if len(data) < 10:
            print("Not enough data for " + pair)
            continue
        data = data[-10:]
        trend, value = botLogic(data, models_1W[index])
        cell_list.append(trend)
        cell_list.append(value)
    updateSheet(wks, 'M10:N21', cell_list)
    print("Updated in google sheet")

if __name__=="__main__":
    # Running jobs once at the start:
    # job1()
    # job2()
    job3()
    job4()
    # time.sleep(50)
    job4a()
    job5()
    # time.sleep(50)
    job6()
    job7()
    # Schedule jobs:
    print("scheduling jobs...")
    # schedule.every(1).minutes.do(job1)
    # schedule.every(15).minutes.do(job2)
    schedule.every(30).minutes.do(job3)
    schedule.every(1).hour.do(job4)
    schedule.every(2).hours.do(job4a)
    schedule.every(4).hours.do(job5)
    schedule.every().day.at("00:01").do(job6)
    schedule.every().monday.at("00:01").do(job7)
    while True:
        now_time = datetime.now(timezone('US/Eastern'))
        day = datetime.now(timezone('US/Eastern')).weekday()
        if(day <= 4):
            print(schedule.jobs)
            schedule.run_pending()
            time.sleep(5)
        else:
            print("Weekend...")
            wks.update('I2', 'Weekend: Bot Paused')
            # sleep for 2 days (1 minute before starting again)
            time.sleep(172740)
            wks.update('I2', '')
            time.sleep(5)
            wks.update('I2', 'Weekend over: Bot Started')
            print("Weekend over Starting again...")