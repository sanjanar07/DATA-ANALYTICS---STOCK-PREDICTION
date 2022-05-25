import numpy as np   #LIBRARIES
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data  #extract data from various Internet sources into a pandas DataFrame
from keras.models import load_model  #we need to load the model that we had created cant run epoch everytime
import streamlit as st

#theming
base="dark"
primaryColor="purple"
font="serif"

start='2010-01-01'   
end='2022-05-20'


#st.title('STOCK TREND PREDICTION')
st.markdown("<h1 style='text-align: center; color: white;'>STOCK TREND PREDICTION</h1>", unsafe_allow_html=True)
st.image('quotes.png')
st.image('image.png')



user_input = st.text_input('Enter Stock Ticker','AMZN')#default is apple#input from user the stock Ticker
df= data.DataReader(user_input,'yahoo',start,end) #scraping data from yahoo finance website 

#Describing data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#visualizations


st.subheader('Closing Price vs Time Chart with 100Ma and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)



# SPLITTING DATA INTO TRAINING AND TESTING- usually done for data predictions

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))# here data scaled down to values between 0 and 1

data_training_array = scaler.fit_transform(data_training)


# DIVIDING DATA INTO x train and y train
x_train=[]#previous 100 days 
y_train=[]#101th day to be predicted 

#we are using simple time series analysis analogy that value for a particular day will be dependent on previous daya(we have defined step as 100 )

for i in range(100,data_training_array.shape[0]):#index 0 indicates 2183
      x_train.append(data_training_array[i-100:i])
      y_train.append(data_training_array[i,0])
 
x_train,y_train=np.array(x_train),np.array(y_train)


#need not do 4 layers and all again just load model

model = load_model('keras_model.h5')

#Testing Part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing],ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0]) 

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


#Final  Graph

st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)



