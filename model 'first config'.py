import numpy as np
import tensorflow as tf
import pandas as pd

data = pd.read_csv('배터리만.csv')

y데이터 = data ['updown'].values

x데이터 = []

for i, rows in data.iterrows():
   x데이터.append([ rows['model name']]) 






model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid'),#마지막 레이어는 하나만 넣어놔야 하나가 나옴
])
 

model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x데이터), np.array(y데이터), epochs=5000)

model.predict()
