import numpy as np
import tensorflow as tf
import pandas as pd

data = pd.read_csv('samsung_stock.csv')


y데이터 = data ['updown'].values

x데이터 = [ ]

for i, rows in data.iterrows():
   x데이터.append([ rows['date'], rows['Open'], rows['Close'], rows['Adj Close']]) 







model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='tanh'),#마지막 레이어는 하나만 넣어놔야 하나가 나옴
])
 

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


model.fit(np.array(x데이터), np.array(y데이터), epochs=100)

예측값=model.predict([[6218, 55700,58100,58100]])
print(예측값)