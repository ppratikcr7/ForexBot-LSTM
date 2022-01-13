#Importing Libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tradermade as tm
from datetime import datetime, timedelta
from pytz import timezone
import pandas as pd

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Load dataset for tradermade:
api_key = "loGqUPLhg0sYLb7NyfSY"
tm.set_rest_api_key(api_key)
time_frames = ['1W']
time = [1]
start_dates = [365]
symbols = ['EURUSD', 'GBPJPY', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD', 'XAUUSD']
# symbols = ['EURUSD']
intervals = ['daily']

for pair in symbols:
    for index, time_frame in enumerate(time_frames):
        start_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=start_dates[index])).strftime("%Y-%m-%d-%H:%M")
        end_date = (datetime.now(timezone('US/Eastern'))).strftime("%Y-%m-%d-%H:%M")
        print(pair, time_frame, start_date, end_date, intervals[index], time[index])
        df = tm.timeseries(currency=pair, start=start_date,end=end_date,interval=intervals[index],fields=["close"],period=time[index])
        df = df['close']
        if time_frame == '4H':
            df = df.iloc[::4]
        elif time_frame == '1W':
            print(start_date, end_date)
            start_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=730)).strftime("%Y-%m-%d-%H:%M")
            end_date = (datetime.now(timezone('US/Eastern')) - timedelta(days=366)).strftime("%Y-%m-%d-%H:%M")
            # print(start_date, end_date)
            df2 = tm.timeseries(currency=pair, start=start_date,end=end_date,interval=intervals[index],fields=["close"],period=time[index])
            df2 = df2['close']
            df = pd.concat([df2, df], ignore_index=True)
            df = df.iloc[::5]
        ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
        scaler=MinMaxScaler(feature_range=(0,1))
        df1=scaler.fit_transform(np.array(df).reshape(-1,1))
        ##splitting dataset into train and test split
        training_size=int(len(df1)*0.65)
        test_size=len(df1)-training_size
        train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
        time_step = 10
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)
        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
        ### Create the Stacked LSTM model
        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=time_step,verbose=1)
        # Save the model
        model.save('models/' + time_frame + '/' + pair + '_' + time_frame + '_model.h5')