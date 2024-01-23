# import library
import streamlit as st, pandas as pd, numpy as np
import plotly.express as px


from vnstock import * #import all functions, including functions that provide OHLC data for charting
from vnstock.chart import * # import chart functions
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# function fetch data to prediction
def df_stock_price_history(ticker, start_date, end_date):
  df = stock_historical_data(symbol=ticker, start_date=str(start_date), end_date=str(end_date), resolution="1D", type="stock", beautify=True, decor=False, source='DNSE')
  df["adj close"] = df["close"]
  df['time'] = pd.to_datetime(df['time'])
  # Set 'time' column as the index
  df.set_index('time', inplace=True)
  # Switch the position of 'Adj Close' and 'Volume'
  df = df[['open', 'high', 'low', 'close', 'adj close', 'volume']]
  # Rename columns
  df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'adj close': 'Adj Close', 'volume': 'Volume'}, inplace=True)
  return df


# Create a new dataframe with only the 'Close column

def create_dataframe(df):
  data = df.filter(['Close'])
  # Convert the dataframe to a numpy array
  dataset = data.values
  # Get the number of rows to train the model on
  training_data_len = int(np.ceil( len(dataset) * .95 ))
  return dataset, training_data_len
  
def scale_data(dataset):
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)
  return scaled_data

def create_training_dataset(dataset, training_data_len, scaled_data):
  
  # Create the scaled training data set
  train_data = scaled_data[0:int(training_data_len), :]
  # Split the data into x_train and y_train data sets
  x_train = []
  y_train = []

  for i in range(60, len(train_data)):
      x_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i, 0])

  # Convert the x_train and y_train to numpy arrays
  x_train, y_train = np.array(x_train), np.array(y_train)

  # Reshape the data
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  return x_train,y_train

# Training model
def train_model(x_train, y_train):
   # Build the LSTM model
  model = Sequential()
  model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
  model.add(LSTM(64, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  # Compile the model
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model.fit(x_train, y_train, batch_size=1, epochs=1)

  return model

# Create the testing data set
def create_testing_dataset(dataset, training_data_len, scaled_data):  
  # Create a new array containing scaled values from index 1543 to 2002
  test_data = scaled_data[training_data_len - 60: , :]
  # Create the data sets x_test and y_test
  x_test = []
  y_test = dataset[training_data_len:, :]
  for i in range(60, len(test_data)):
      x_test.append(test_data[i-60:i, 0])

  # Convert the data to a numpy array
  x_test = np.array(x_test)

  # Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
  return x_test,y_test

# Prediction
def prediction(model, scaler, x_test, y_test):
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  # Get the root mean squared error (RMSE)
  rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
  return predictions, rmse

# What was the moving average of the various stocks ?
def get_moving_avg_fig(df,st):
  ma_day = [10,20,50]
  data = df.copy()
  data.drop(['Volume', 'Open', 'High', 'Close', 'Low'], axis=1, inplace=True)
  for ma in ma_day:
    column_name = f"MA for {ma} day"
    data[column_name] = data["Adj Close"].rolling(ma).mean()
  fig = px.line(data,x=data.index, y=data.columns, title="Moving average of: " + ticker, template="plotly_dark")
  st.plotly_chart(fig)

# What was the daily return of the stock on average ?
def get_daily_return_fig(df,st):
  data = df.copy()
  data["Daily Return"] = data["Adj Close"].pct_change()
  fig = px.line(data,x=data.index, y=data["Daily Return"], title="Daily return of: " + ticker, template="plotly_dark")
  st.plotly_chart(fig)


# Get list companies from vnstock
list_com = listing_companies(live=True)
# Companies DataFrame
df_companies = pd.DataFrame(list_com)
# Get list Companies's name
list_companies_name = df_companies["ticker"]


st.title('Stock Dashboard')
ticker = st.sidebar.selectbox("Ticker", list_companies_name)
start_date = st.sidebar.date_input('Start Date', (datetime.now() - timedelta(days=360)).date())
end_date = st.sidebar.date_input('End Date')


# Fetch data from user input using vnstock library
data = df_stock_price_history(ticker, start_date, end_date)


# Plot the data
st.header("Data Visualization")
st.subheader("Plot of the data")
st.write("Data from ", start_date, " to ", end_date)
fig = px.line(data,x=data.index, y=data.columns, title="Closing price of: " + ticker, template="plotly_dark")
st.plotly_chart(fig)

st.subheader("Moving average")
get_moving_avg_fig(data,st)

st.subheader("Daily return")
get_daily_return_fig(data,st)

st.subheader("Predicting the closing price")
if st.button("Prediction"):
  data_pred = df_stock_price_history(ticker,"2000-01-01", datetime.now().date())
  dataset, training_data_len = create_dataframe(data_pred)
  scaled_data = scale_data(dataset)
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  x_train,y_train = create_training_dataset(dataset,training_data_len,scaled_data)
  model = train_model(x_train,y_train)
  x_test,y_test = create_testing_dataset(dataset,training_data_len,scaled_data)
  predictions, rmse = prediction(model,scaler,x_test,y_test)
  # st.write(rmse)

  train = data_pred[:training_data_len]
  valid = data_pred[training_data_len:]
  valid['Predictions'] = predictions
  # Visualize the data
  # fig_plt = plt.figure(figsize=(16,6))
  # plt.title('Model')
  # plt.xlabel('Date', fontsize=18)
  # plt.ylabel('Close Price USD ($)', fontsize=18)
  # plt.plot(train['Close'])
  # plt.plot(valid[['Close', 'Predictions']])
  # plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  # st.pyplot(fig_plt)
  valid.drop(['Volume', 'Open', 'High', 'Close', 'Low'], axis=1, inplace=True)
  fig_pred = px.line(valid,x=valid.index, y=valid.columns, title="Predicting the closing price stock price of " + ticker, template="plotly_dark")
  st.plotly_chart(fig_pred)




