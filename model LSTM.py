import numpy as np
import tensorflow as tf
import pandas as pd

# Load data
data = pd.read_csv('samsungint.csv')

# Extract targets
y데이터 = data['updown'].values

# Extract features
x데이터 = []
for i, rows in data.iterrows():
   x데이터.append([rows['date'], rows['Open'], rows['Close'], rows['Adj Close']])
x데이터 = np.array(x데이터)  # Convert list to NumPy array

# Reshape data to the required dimensions: (samples, timesteps, features)
x데이터 = x데이터.reshape((x데이터.shape[0], 1, x데이터.shape[1]))  # Example reshaping

# Build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(1, 4)))
model.add(tf.keras.layers.Dense(1, activation = 'relu'))

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Fit model
model.fit(x데이터, y데이터, epochs=200, verbose=1)

# Predict
predicted_output = model.predict(x데이터[1000].reshape(1, 1, 4))  # Adjust shape for prediction
print("예측 값:", predicted_output)
