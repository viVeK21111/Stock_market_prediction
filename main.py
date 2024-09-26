import streamlit as st 
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpaperaccess.com/full/1393758.jpg");
        background-size: cover;
    }
    .stApp > div {
        backdrop-filter: blur(9px);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
       padding:15px;
       background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent black background */
       border-radius:10px;
       color:white;
       font-size:large;
       height: auto; /* Adjust height as needed */
    }
    [data-testid="stHeader"] {
        background-color:rgba(0,0,0,0);
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Google Stock Price Prediction")
st.write("Predict future Google stock prices using LSTM and technical indicators.")

model = load_model('models/modelt.keras')
scaler1 = joblib.load('models/scaler1.pkl')
scaler2 = joblib.load('models/scaler2.pkl')
scaler3 = joblib.load('models/scaler3.pkl')
scaler4 = joblib.load('models/scaler4.pkl')
scaler5 = joblib.load('models/scaler5.pkl') 
datem = dt.date.today() - relativedelta(months=5)
temp = yf.download("GOOGL",datem,dt.date.today())
timestamps=100+20
temp=temp[-timestamps:]

datagn = pd.DataFrame()
datagn['MA_10'] = temp['Adj Close'].rolling(window=10).mean()
datagn['Volatility'] = temp['Adj Close'].rolling(window=10).std()
datagn['Daily Return'] = temp['Adj Close'].pct_change()
datagn['Volume Change'] = temp['Volume'].rolling(window=10).mean()

def calculate_rsi(data, period):
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
period =20
datagn['RSI'] = calculate_rsi(temp,period)
datagn['Adj Close']=temp['Adj Close']
datagn['MA_10'] = scaler1.fit_transform(datagn['MA_10'].values.reshape(-1,1))
datagn['Adj Close'] = scaler2.fit_transform(datagn['Adj Close'].values.reshape(-1,1))
datagn['Volatility']=scaler3.fit_transform(datagn['Volatility'].values.reshape(-1,1))
datagn['RSI'] = scaler4.fit_transform(datagn['RSI'].values.reshape(-1,1))
datagn['Volume Change'] = scaler5.fit_transform(datagn['Volume Change'].values.reshape(-1,1)) 
datagn=datagn.dropna()
datagn= datagn.values
datagn = datagn.reshape(1,datagn.shape[0],datagn.shape[1])
days=3
y_pred=model.predict(datagn)
st.write(model,y_pred,datagn.shape)
ele = scaler2.inverse_transform(y_pred)
ldays = st.number_input("Select number of past days to display", min_value=5, max_value=15, value=7, step=1)

dff1 = temp['Adj Close'][-ldays:] # last ten days
date_index = pd.date_range(start=dt.date.today(), periods=days, freq='D')
dff = pd.DataFrame(index=date_index)
dff['Adj Close']=ele[0]
dff = dff.rename_axis('Date')
dff2 = pd.concat([dff1,dff],axis=0)

combined_df = dff2
plt.figure(figsize=(12, 6))
plt.plot(combined_df.index, combined_df['Adj Close'], marker='o', label='Close Prices')
plt.axvline(x=combined_df.index[-(days)], color='red', linestyle='--', label='Prediction Start')  # Line to indicate where predictions start
plt.title('Google Stock Price')
plt.xlabel('Date')
plt.ylabel('Adj Closing Price')
plt.xticks(ticks=combined_df.index, rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
#plt.show()
st.pyplot(plt)