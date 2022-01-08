# ForexBot
TraderMade based ForexBot Script for Future Price Move Prediction:

Download and install python 3.7.9: (https://www.python.org/downloads/)
mac 64bit: https://www.python.org/ftp/python/3.7.9/python-3.7.9-macosx10.9.pkg
windows 64bit: https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe
linux 64bit: https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz

Install pip:
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

Install jupyter:
pip3 install jupyter

Setup virtual environment separate for your project:
pip3.7 install virtualenv
virtualenv -p python3 forexbot or
python3.7 -m virtualenv forexbot
source forexbot/bin/activate
pip3 install -r requirements.txt
(created using pip3 freeze > requirements.txt)

Run program:
1. First edit ForexModelTrainer.py and ForexBot.py files and put you api_key for tradermade account.
2. Run the Forex bot that will update the google sheet (https://docs.google.com/spreadsheets/d/1rGRqOwICVfW8TEpgoiNN6KhbDfGKBF-z0Id8gcAOkQ4/edit#gid=631979319) named "Final" with predictions for time frames according to their time frame.
3. Run the trainer Bot every week on weekends to get new trend to be learnt by the LSTM bot in the new trained models for each time frames: python3 ForexModelTrainer.py
4. There one more jupyter notebook "Forex_Prediction_CandleStick_Analysis.ipynb" which has initial level candlestick pattern analysis and basic LSTM model with overall accuracy of various predictions.
 