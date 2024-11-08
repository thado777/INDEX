import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
data = pd.read_csv('samsung_stock.csv')

# y데이터 설정
y데이터 = data['updown'].values

# x데이터 설정 및 날짜를 숫자로 변환
data['date'] = pd.to_datetime(data['date'])
data['date_numeric'] = data['date'].astype(np.int64) // 10**9  # 날짜를 초 단위 타임스탬프로 변환
x데이터 = data[['date_numeric', 'Open', 'Close', 'Adj Close']].values

# 데이터 정규화
scaler = MinMaxScaler()
x데이터 = scaler.fit_transform(x데이터)

# 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(x데이터.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid'),  # 활성화 함수를 sigmoid로 변경
])

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(x데이터, y데이터, epochs=100, batch_size=32, validation_split=0.2)

# 예측 수행 (입력값을 정규화해야 함)
예측값 = model.predict(scaler.transform([[6218, 55700, 58100, 58100]]))
print(예측값)
