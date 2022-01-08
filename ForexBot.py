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

n_steps=100
scaler=MinMaxScaler(feature_range=(0,1))
symbols = ['EURUSD', 'GBPJPY', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD', 'XAUUSD']
api_key = ""
tm.set_rest_api_key(api_key)

# Google Sheet:
sa = gspread.service_account()
sh = sa.open("FX - AI/ML Model Sheet")
wks = sh.worksheet("Final")

# Load all trained models once each week:
models_1M = []
print("Loading 1M models...")
for index, pair in enumerate(symbols):
    model = load_model('models/1M/'+ pair + '_1M_model.h5')
    models_1M.append(model)

models_15M = []
print("Loading 15M models...")
for index, pair in enumerate(symbols):
    model = load_model('models/15M/'+ pair + '_15M_model.h5')
    models_15M.append(model)

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

models_1D = []
print("Loading 1D models...")
for index, pair in enumerate(symbols):
    model = load_model('models/1D/'+ pair + '_1D_model.h5')
    models_1D.append(model)

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
def updateSheet(wks, cell1, cell2, trend, expected_move): 
    wks.update(cell1, trend)
    wks.update(cell2, expected_move)

def botLogic(df, model, col1, col2, cell):
    df=scaler.fit_transform(np.array(df).reshape(-1,1))
    lst = []
    df = df[:100, 0]
    lst.append(df)
    df = np.array(lst)
    pred = predictionFunction(df, model)
    last_val = scaler.inverse_transform(df[-1].reshape(1,-1))
    trend, expected_move = transformValue(last_val, pred)
    updateSheet(wks, col1+cell, col2+cell, trend, json.dumps(np.round(expected_move.astype(float),5)))
    print("Updated in google sheet")

def job1():
    print("1 min...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=1)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD', start=from_date,end=to_datetime,interval="minute",fields=["close"],period=1)
    for index, pair in enumerate(symbols):
        data = df[pair]
        botLogic(data, models_1M[index], 'C', 'D', str(index+10))

def job2():
    print("15 min...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=2)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD', start=from_date,end=to_datetime,interval="minute",fields=["close"],period=15)
    for index, pair in enumerate(symbols):
        data = df[pair]
        botLogic(data, models_15M[index], 'E', 'F', str(index+10))

def job3():
    print("30 min...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=3)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD', start=from_date,end=to_datetime,interval="minute",fields=["close"],period=30)
    for index, pair in enumerate(symbols):
        data = df[pair]
        botLogic(data, models_30M[index], 'G', 'H', str(index+10))

def job4():
    print("1 hour...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=7)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD', start=from_date,end=to_datetime,interval="hourly",fields=["close"],period=1)
    for index, pair in enumerate(symbols):
        data = df[pair]
        botLogic(data, models_1H[index], 'I', 'J', str(index+10))

def job5():
    print("1 day...")
    from_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=150)).strftime("%Y-%m-%d-%H:%M")
    to_datetime = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
    df = tm.timeseries(currency='EURUSD,GBPJPY,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,XAUUSD', start=from_date,end=to_datetime,interval="daily",fields=["close"],period=1)
    for index, pair in enumerate(symbols):
        data = df[pair]
        botLogic(data, models_1D[index], 'K', 'L', str(index+10))

if __name__=="__main__":
    # Running jobs once at the start:
    job1()
    job2()
    job3()
    job4()
    job5()
    # Schedule jobs:
    print("scheduling jobs...")
    schedule.every(1).minutes.do(job1)
    schedule.every(15).minutes.do(job2)
    schedule.every(30).minutes.do(job3)
    schedule.every(1).hour.do(job4)
    schedule.every().day.at("00:01").do(job5)

    while True:
        schedule.run_pending()
        time.sleep(1)