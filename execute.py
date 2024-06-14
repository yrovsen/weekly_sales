import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load
from lightgbm import LGBMRegressor, Booster
from datetime import datetime


model = Booster(model_file='best_model.txt')


data = pd.read_csv('car_last.csv')

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
df = data.copy()

data['day'] = data['Date'].dt.day
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year

data.drop(columns = ['Date'], inplace = True)

def predict(list_):
    y_pred = model.predict([list_])
    return y_pred

def main():
    st.title('Weekly Sales Prediction')
    st.write('Enter values according to given statements:')
    
    
    store = st.number_input('Store', min_value=0, max_value=int(data['Store'].max()), step=int(data['Store'].min()), value=0)
    holiday_flag = st.selectbox('Holiday Flag', ['Yes', 'No'])

    if holiday_flag == 'Yes':
        category_encoded = 1
    else:
        category_encoded = 0

    cpi = st.number_input('CPI', min_value=float(data['CPI'].min()), max_value=float(data['CPI'].max()), step=float(data['CPI'].min()), value=float(data['CPI'].min()))
    
    date = st.date_input("Select a date", min_value=df['Date'].min(), max_value=df['Date'].max(), value = df['Date'].min())
    day = date.day
    month = date.month
    year = date.year
    
    list_ = [store, category_encoded, cpi, day, month, year]
    if st.button('Predict'):
        predicted_value = predict(list_)
        st.write(f'Predicted price for given values is {predicted_value[0].round(2)}  AZN')
if __name__ == '__main__':
    main()