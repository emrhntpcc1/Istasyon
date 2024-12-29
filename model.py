import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
df = pd.concat([df1,df2])



#Bu aykırı değerleri bulalım
Q1 = df["yolcu_sayisi"].quantile(0.25)
Q3 = df["yolcu_sayisi"].quantile(0.75)
IQR = (Q3 - Q1) * 1.5

#print(f"Q1: {Q1}")
#print(f"Q3: {Q3}")
#print(f"IQR: {IQR}")

#print("------------------------------------")
alt_sinir = Q1 - IQR
ust_sinir = Q3 + IQR

#print(f"Alt Sinir: {alt_sinir}")
#print(f"Üst Sinir: {ust_sinir}")

#print("------------------------------------")
#Alt sinir değeri negatif olamaz bundan dolayı altsinir i 0 yapmalıyız.
alt_sinir = 0
#print(f"Yeni Alt Sinir: {alt_sinir}")

#Aykırı değerleri bulalım.
df_yolcu_sayisi = df["yolcu_sayisi"]

#print("Aykırı Değerler")
aykiri_degerler = df_yolcu_sayisi[df_yolcu_sayisi > ust_sinir]
#print(aykiri_degerler)

# Veri setinden aykırı değerleri temizle
df_temiz = df[df["yolcu_sayisi"] <= ust_sinir]

#print(f"Temiz olmayan veri: {df.shape}")
#print(f"Temizlenmiş veri: {df_temiz.shape}")

# Temizlenmiş veriyi orjinal veri setine de aktardım.
df = df_temiz

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Tarih bilgilerini ayrıştırma ve veri hazırlığı
df['yıl'] = pd.to_datetime(df['yolculuk_tarihi']).dt.year
df['ay'] = pd.to_datetime(df['yolculuk_tarihi']).dt.month
df['gün'] = pd.to_datetime(df['yolculuk_tarihi']).dt.day
df['saat'] = df['saat_sin']  # 'saat_sin' ve 'saat_cos' özellikleri zaten var

# Kullanılacak özellikler
features = df[['yıl', 'ay', 'gün', 'saat_sin', 'saat_cos', 'tatil_gunu_mu',
               'Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']]
target = df['yolcu_sayisi']

# Normalizasyon
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(features)
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Zaman serisi verisi oluşturma
def create_sequences(data, target, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24
X, y = create_sequences(scaled_features, scaled_target, seq_length)

# Eğitim ve test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM modelini oluştur
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Modeli derle
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Modeli eğit
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Modeli değerlendirme
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Tahmin için fonksiyon
def predict_passenger_count_LSTM(year, month, day, hour_sin, hour_cos, holiday, week_days):
    """
    year: Yıl bilgisi (ör. 2024)
    month: Ay bilgisi (1-12)
    day: Gün bilgisi (1-31)
    hour_sin: Saatin sinüs dönüşümü
    hour_cos: Saatin kosinüs dönüşümü
    holiday: Tatil günü mü? (1: Evet, 0: Hayır)
    week_days: Haftanın günü bilgileri [Pazartesi, Salı, ..., Pazar] (binary liste)
    """
    input_data = np.array([[year, month, day, hour_sin, hour_cos, holiday] + week_days])
    scaled_input = scaler_features.transform(input_data)
    sequence = np.expand_dims(scaled_input, axis=1)
    prediction = model.predict(sequence)
    return scaler_target.inverse_transform(prediction)


#Saat için fonskiyon

model.save("model_LSTM.h5")

import math

def hour_features(hour):
    """
    Saat bilgisini alır ve sinüs/kosinüs dönüşümü yapar.
    Bu dönüşümler döngüsel zaman bilgilerini modele daha iyi aktarır.

    Parametre:
        hour (int): 0 ile 23 arasında bir saat değeri.

    Dönen:
        (tuple): Saatin sinüs ve kosinüs dönüşümünden oluşan tuple.
    """
    # Saatten açıya çevirme (0-23 saat → 0-360 dereceye)
    hour_angle = (hour % 24) * 15  # 360° / 24 = 15°

    # Sinüs ve kosinüs dönüşümü
    hour_sin = math.sin(math.radians(hour_angle))
    hour_cos = math.cos(math.radians(hour_angle))

    return hour_sin, hour_cos

#LSTM Modeli tahmini
#Holidaydan kasttımız resmi tatil olması

hour_sin, hour_cos = hour_features(14)
x = predict_passenger_count_LSTM(
    year=2024,
    month=9,
    day=4,
    hour_sin=hour_sin,
    hour_cos=hour_cos,
    holiday=0,
    week_days=[0, 0, 1, 0, 0, 0, 0]
)
print(f"Tahmin edilen yolcu sayısı: {x[0][0]:.2f}")
