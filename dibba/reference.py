import numpy as np
import pandas as pd
import seaborn as sns
import keras
import sqlite3 
from sqlite3 import Error
import os
import django
from django.conf import settings
import sys
from datetime import datetime

# sys.path = sys.path + ['E:\FINAL CMS\salesandinventoryCMS (1)\salesandinventoryCMS']
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'InventoryMS.settings')
# django.setup()
# from models import Prediction

def create_connection():
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
    except Error as e:
        print(e)

    return conn


# This method creates the necessary tables in the database
def create_table(conn):
    curr = conn.cursor()
    curr.execute("""DROP TABLE IF EXISTS prediction""")
    curr.execute('''CREATE TABLE prediction(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date datetime,
            price real
            )''')


DB_PATH = r'E:\FINAL CMS\salesandinventoryCMS (1)\salesandinventoryCMS\db.sqlite3'
conn = create_connection()
create_table(conn)   

#This method stores into database
def insert_data(conn, date, price):
    curr = conn.cursor()
    curr.execute("INSERT INTO prediction (date, price) VALUES (?, ?)", (date, price))
    conn.commit()
        
sns.set(style="darkgrid")
from datetime import datetime, timedelta,date
items=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/items.csv")
shops=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/shops.csv")
cats=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/item_categories.csv")
test=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/test - Copy.csv")
train=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/sales_train.csv")

import tensorflow as tf

# Display TensorFlow version
print("TensorFlow version:", tf.__version__)

# Display Keras version
print("Keras version:", keras.__version__)

# Check for Outliers
# plt.figure(figsize=(10,4))

flierprops = dict(marker='o', markerfacecolor='purple', markersize=6,
                  linestyle='none', markeredgecolor='black')
sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)

# plt.figure(figsize=(10,4))
# plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price, flierprops=flierprops)

# Fixing a threshold based on above figures-
# eliminate item_cnt_day more than 1000 in one day
# eliminate item_price more than 100000
train = train[(train.item_price < 100000 )& (train.item_cnt_day < 1000)]
# also removing all the negative item prices
train = train[(train.item_price > 0) & (train.item_cnt_day > 0)].reset_index(drop = True)
train.head()

train.shape
print(train)


train['date'] = pd.to_datetime(train['date'], format="%d.%m.%Y")

#represent month in date field as its first day
train['date'] = train['date'].dt.year.astype('str') + '-' + train['date'].dt.month.astype('str') + '-01'
train['date'] = pd.to_datetime(train['date'])
#groupby date and sum the sales
train = train.groupby('date').item_cnt_day.sum().reset_index()


train.rename(columns = {'item_cnt_day':'item_cnt_month'}, inplace = True)

df_sales_copy= train.copy()

df_sales_copy=df_sales_copy.set_index('date')

df_diff = train.copy()
#add previous sales to the next row
df_diff['prev_item_cnt_month'] = df_diff['item_cnt_month'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['item_cnt_month'] - df_diff['prev_item_cnt_month'])
df_diff.head(10)

#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_item_cnt_month'],axis=1)
#adding lags
for inc in range(1,10):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)
df_supervised.head()

#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['item_cnt_month','date'],axis=1)
#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


from keras.models import load_model

# Load the model
model = load_model('C:/Users/ankur/OneDrive/Desktop/dibba/saved_model.hdf5')

print('X_test:', X_test)
y_pred = model.predict(X_test,batch_size=1)

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print (np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(train[-10:].date)
act_sales = list(train[-10:].item_cnt_month)
price_list=[]
date_list=[]
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index+1]
    
    price_list.append(int(pred_test_set_inverted[index][0] + act_sales[index]))
    date_list.append(sales_dates[index+1])
    pred_date=datetime.strptime(str(sales_dates[index+1]), '%Y-%m-%d %H:%M:%S')
    formatted_date = pred_date.strftime('%Y-%m-%d %H:%M:%S')
    insert_data(conn,formatted_date,int(pred_test_set_inverted[index][0] + act_sales[index]))
    result_list.append(result_dict)
    
df_result = pd.DataFrame(result_list)
print(df_result)
#for multistep prediction, replace act_sales with the predicted sales

df_result_copy= df_result.copy()
df_result_copy=df_result_copy.set_index('date')
print("copy")


# fig = plt.figure(figsize=[20,5])
# fig.suptitle('sales')
# Actual, = plt.plot(df_sales_copy.index, df_sales_copy, 'b.-', label='Actual')
# predicted, = plt.plot(df_result_copy.index, df_result_copy, 'g.-', label='Predicted')
# plt.legend(handles=[Actual,predicted])
# plt.show()
