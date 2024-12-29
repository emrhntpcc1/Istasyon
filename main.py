import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
import os
import datetime
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
from ttkbootstrap.widgets import Progressbar
from PIL import Image, ImageTk


# ===[Model]===
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


model = load_model("model_LSTM.h5", custom_objects={'mse': mse})

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
df = pd.concat([df1, df2])

df['yıl'] = pd.to_datetime(df['yolculuk_tarihi']).dt.year
df['ay'] = pd.to_datetime(df['yolculuk_tarihi']).dt.month
df['gün'] = pd.to_datetime(df['yolculuk_tarihi']).dt.day
df['saat'] = df['saat_sin']

features = df[['yıl', 'ay', 'gün', 'saat_sin', 'saat_cos', 'tatil_gunu_mu', 'Pazartesi', 'Salı', 'Çarşamba', 'Perşembe',
               'Cuma', 'Cumartesi', 'Pazar']]
target = df['yolcu_sayisi']

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(features)
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))


# ===[ARAYÜZ]===
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def sayfayiGoster(sayfa):
    sayfa.tkraise()


def arayuzOlustur():
    root = tk.Tk()
    root.title("İstasyon")
    root.iconbitmap("logoson.ico")
    root.geometry("400x600")

    # Sayfa çerçeveleri
    anaSayfa_frame = tk.Frame(root)
    m1bilgi_frame = tk.Frame(root)
    m2bilgi_frame = tk.Frame(root)
    m3bilgi_frame = tk.Frame(root)
    m4bilgi_frame = tk.Frame(root)
    m5bilgi_frame = tk.Frame(root)
    m6bilgi_frame = tk.Frame(root)
    m7bilgi_frame = tk.Frame(root)
    m8bilgi_frame = tk.Frame(root)
    m9bilgi_frame = tk.Frame(root)
    hakkinda_frame = tk.Frame(root)
    hesapla_frame = tk.Frame(root)
    yukleme_frame = tk.Frame(root)

    for frame in (anaSayfa_frame, m1bilgi_frame, m2bilgi_frame, m3bilgi_frame, m4bilgi_frame, m5bilgi_frame,
                  m6bilgi_frame, m7bilgi_frame, m8bilgi_frame, m9bilgi_frame, hakkinda_frame, hesapla_frame,
                  yukleme_frame):
        frame.grid(row=0, column=0, sticky="nsew")

    # ===[Yükleme Ekranı]===
    yukleme_image = ImageTk.PhotoImage(Image.open("yukleme.jpg").resize((400, 600), Image.Resampling.LANCZOS))
    canvas = tk.Canvas(yukleme_frame, width=400, height=600, highlightthickness=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=yukleme_image)

    # ===[Anasayfa]===
    anaSayfa_image = ImageTk.PhotoImage(Image.open("anaSayfa.png").resize((400, 600), Image.Resampling.LANCZOS))
    canvas = tk.Canvas(anaSayfa_frame, width=400, height=600, highlightthickness=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=anaSayfa_image)

    canvas.create_rectangle(20, 250, 180, 450, outline="", tags="hesapla")
    canvas.create_rectangle(220, 250, 380, 450, outline="", tags="bilgi")

    canvas.tag_bind("hesapla", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("bilgi", "<Button-1>", lambda event: sayfayiGoster(m1bilgi_frame))

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="rota")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="metro")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")

    canvas.tag_bind("rota", "<Button-1>", lambda event: sayfayiGoster(m1bilgi_frame))
    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))

    # ===[Bilgi Ekranı(1)]===
    m1bilgi_image = ImageTk.PhotoImage(Image.open("m1Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m1bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m1bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m2bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m9bilgi_frame))

    # ===[Bilgi Ekranı(2)]===
    m2bilgi_image = ImageTk.PhotoImage(Image.open("m2Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m2bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m2bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m3bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m1bilgi_frame))

    # ===[Bilgi Ekranı(3)]===
    m3bilgi_image = ImageTk.PhotoImage(Image.open("m3Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m3bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m3bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m4bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m2bilgi_frame))

    # ===[Bilgi Ekranı(4)]===
    m4bilgi_image = ImageTk.PhotoImage(Image.open("m4Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m4bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m4bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m5bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m3bilgi_frame))

    # ===[Bilgi Ekranı(5)]===
    m5bilgi_image = ImageTk.PhotoImage(Image.open("m5Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m5bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m5bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m6bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m4bilgi_frame))

    # ===[Bilgi Ekranı(6)]===
    m6bilgi_image = ImageTk.PhotoImage(Image.open("m6Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m6bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m6bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m7bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m5bilgi_frame))

    # ===[Bilgi Ekranı(7)]===
    m7bilgi_image = ImageTk.PhotoImage(Image.open("m7Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m7bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m7bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m8bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m6bilgi_frame))

    # ===[Bilgi Ekranı(8)]===
    m8bilgi_image = ImageTk.PhotoImage(Image.open("m8Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m8bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m8bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m9bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m7bilgi_frame))

    # ===[Bilgi Ekranı(9)]===
    m9bilgi_image = ImageTk.PhotoImage(Image.open("m9Bilgi.png").resize((400, 600),
                                                                        Image.Resampling.LANCZOS))
    canvas = tk.Canvas(m9bilgi_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=m9bilgi_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="metro")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")
    canvas.create_rectangle(350, 230, 387, 290, outline="", tags="ileri")
    canvas.create_rectangle(8, 230, 45, 290, outline="", tags="geri")

    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("info", "<Button-1>", lambda event: sayfayiGoster(hakkinda_frame))
    canvas.tag_bind("ileri", "<Button-1>", lambda event: sayfayiGoster(m1bilgi_frame))
    canvas.tag_bind("geri", "<Button-1>", lambda event: sayfayiGoster(m8bilgi_frame))

    # ===[Hakkında Ekranı]===
    hakkinda_image = ImageTk.PhotoImage(Image.open("hakkinda.png").resize((400, 600),
                                                                          Image.Resampling.LANCZOS))
    canvas = tk.Canvas(hakkinda_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=hakkinda_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="rota")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="metro")

    canvas.tag_bind("rota", "<Button-1>", lambda event: sayfayiGoster(m1bilgi_frame))
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: sayfayiGoster(anaSayfa_frame))
    canvas.tag_bind("metro", "<Button-1>", lambda event: sayfayiGoster(hesapla_frame))

    # ===[Hesaplama Ekranı]===
    def tahminToBilgi():
        ongoru_sil(canvas)
        sayfayiGoster(m1bilgi_frame)
        progress["value"] = 0
        canvas.itemconfig("progress_text", text=f"{0}%")

    def tahminToAnasayfa():
        ongoru_sil(canvas)
        sayfayiGoster(anaSayfa_frame)
        progress["value"] = 0
        canvas.itemconfig("progress_text", text=f"{0}%")

    def tahminToHakkinda():
        ongoru_sil(canvas)
        sayfayiGoster(hakkinda_frame)
        progress["value"] = 0
        canvas.itemconfig("progress_text", text=f"{0}%")

    hesaplamaSayfa_image = ImageTk.PhotoImage(Image.open("hesaplamaSayfasi.png").resize((400, 600),
                                                                                        Image.Resampling.LANCZOS))

    canvas = tk.Canvas(hesapla_frame, width=400, height=600, highlightthickness=0, borderwidth=0, bg='#333333')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=hesaplamaSayfa_image)

    canvas.create_rectangle(60, 540, 120, 590, outline="", tags="rota")
    canvas.create_rectangle(160, 540, 220, 590, outline="", tags="anasayfa")
    canvas.create_rectangle(260, 540, 320, 590, outline="", tags="info")

    canvas.create_text(310, 246, text="0%", font=("Helvetica", 10, "bold"), fill="white", tags="progress_text")
    canvas.create_text(185, 230, text="Yoğunluk", font=("Helvetica", 12, "bold"), fill="gray")

    canvas.tag_bind("rota", "<Button-1>", lambda event: tahminToBilgi())
    canvas.tag_bind("anasayfa", "<Button-1>", lambda event: tahminToAnasayfa())
    canvas.tag_bind("info", "<Button-1>", lambda event: tahminToHakkinda())

    # Tarih seçme butonu
    my_date = tb.DateEntry(hesapla_frame, bootstyle="danger", firstweekday=0)
    my_date.place(x=70, y=100)

    # Saat seçme butonu
    saat_combobox = ttk.Combobox(hesapla_frame, values=[f"{i:02d}:00" for i in range(6, 24)], width=5, state="readonly",
                                 style="danger")
    saat_combobox.current(0)
    saat_combobox.place(x=235, y=102)

    # Tatil seçme butonu
    canvas.create_text(182, 150, text="Resmi tatil ise kutucuğu işaretleyiniz",
                       font=("Helvetica", 8, "bold"), fill="white")
    tatil_var = tk.IntVar(value=0)
    tatil_checkbox = tk.Checkbutton(hesapla_frame, text="", variable=tatil_var)
    tatil_checkbox.place(x=290, y=139)

    # Hat seçme butonu
    hat_combobox = ttk.Combobox(hesapla_frame, values=["M1"],
                                width=3, state="readonly", style="danger")
    hat_combobox.current(0)
    hat_combobox.place(x=24, y=101)

    # Yoğunluk çubuğu
    progress = Progressbar(hesapla_frame, length=220, bootstyle="primary", value=0)
    progress.place(x=70, y=240)

    # Tahmin
    def hour_features(hour):
        hour_angle = (hour % 24) * 15
        hour_sin = math.sin(math.radians(hour_angle))
        hour_cos = math.cos(math.radians(hour_angle))
        return hour_sin, hour_cos

    def hesapla():
        tarih = my_date.entry.get()
        day, month, year = map(int, tarih.split("."))
        date = datetime.date(year, month, day)
        hafta_gunu_index = date.weekday()

        if hafta_gunu_index == 0:
            gun = [1, 0, 0, 0, 0, 0, 0]
        elif hafta_gunu_index == 1:
            gun = [0, 1, 0, 0, 0, 0, 0]
        elif hafta_gunu_index == 2:
            gun = [0, 0, 1, 0, 0, 0, 0]
        elif hafta_gunu_index == 3:
            gun = [0, 0, 0, 1, 0, 0, 0]
        elif hafta_gunu_index == 4:
            gun = [0, 0, 0, 0, 1, 0, 0]
        elif hafta_gunu_index == 5:
            gun = [0, 0, 0, 0, 0, 1, 0]
        else:
            gun = [0, 0, 0, 0, 0, 0, 1]

        saat = saat_combobox.get()
        sadece_saat = int(saat.split(":")[0])
        tatil = tatil_var.get()
        oneri_saat1 = sadece_saat - 2
        if oneri_saat1 == 4 or 5:
            oneri_saat1 = sadece_saat

        oneri_saat2 = sadece_saat - 1
        if oneri_saat2 == 5:
            oneri_saat2 = sadece_saat

        oneri_saat3 = sadece_saat + 1
        if oneri_saat3 == 24:
            oneri_saat3 = sadece_saat

        oneri_saat4 = sadece_saat + 2
        if oneri_saat4 == 24 or 25:
            oneri_saat4 = sadece_saat

        hour_sin, hour_cos = hour_features(sadece_saat)
        hour_sin1, hour_cos1 = hour_features(oneri_saat1)
        hour_sin2, hour_cos2 = hour_features(oneri_saat2)
        hour_sin3, hour_cos3 = hour_features(oneri_saat3)
        hour_sin4, hour_cos4 = hour_features(oneri_saat4)

        tahmin = predict_passenger_count_LSTM(
            year=year,
            month=month,
            day=day,
            hour_sin=hour_sin,
            hour_cos=hour_cos,
            holiday=tatil,
            week_days=gun,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            model=model
        )
        print(f"Tahmin edilen yolcu sayısı: {tahmin[0][0]:.2f}")

        oneri1 = predict_passenger_count_LSTM(
            year=year,
            month=month,
            day=day,
            hour_sin=hour_sin1,
            hour_cos=hour_cos1,
            holiday=tatil,
            week_days=gun,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            model=model
        )

        oneri2 = predict_passenger_count_LSTM(
            year=year,
            month=month,
            day=day,
            hour_sin=hour_sin2,
            hour_cos=hour_cos2,
            holiday=tatil,
            week_days=gun,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            model=model
        )

        oneri3 = predict_passenger_count_LSTM(
            year=year,
            month=month,
            day=day,
            hour_sin=hour_sin3,
            hour_cos=hour_cos3,
            holiday=tatil,
            week_days=gun,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            model=model
        )

        oneri4 = predict_passenger_count_LSTM(
            year=year,
            month=month,
            day=day,
            hour_sin=hour_sin4,
            hour_cos=hour_cos4,
            holiday=tatil,
            week_days=gun,
            scaler_features=scaler_features,
            scaler_target=scaler_target,
            model=model
        )

        oneriler = [int(oneri1), int(oneri2), int(oneri3), int(oneri4)]
        en_kucuk = min(oneriler)
        en_kucuk_indeks = oneriler.index(en_kucuk)

        if en_kucuk_indeks == 0:
            son_oneri_saat = oneri_saat1
        elif en_kucuk_indeks == 1:
            son_oneri_saat = oneri_saat2
        elif en_kucuk_indeks == 2:
            son_oneri_saat = oneri_saat3
        else:
            son_oneri_saat = oneri_saat4

        if en_kucuk >= int(tahmin):
            en_kucuk = int(tahmin)
            son_oneri_saat = sadece_saat

        saat_str = str(son_oneri_saat).zfill(2)
        saat_formatli = f"{saat_str}:00"

        canvas.create_text(147, 280, text="Öngörülen Yolcu Sayısı: ",
                           font=("Helvetica", 10, "bold"), fill="white", tags="ongoru_yazi")
        canvas.create_text(303, 280, text="",
                           font=("Helvetica", 10, "bold"), fill="#D9534F", tags="ongoru")
        canvas.itemconfig("ongoru", text=f"{int(tahmin)}")

        canvas.create_text(160, 360, text="Yolculuk İçin Önerilen Saat: ",
                           font=("Helvetica", 10, "bold"), fill="white", tags="oneri_yazi1")

        canvas.create_text(303, 360, text=f"{saat_formatli}",
                           font=("Helvetica", 10, "bold"), fill="white", tags="oneri_saat")

        canvas.create_text(170, 390, text="Önerilen Saatteki Yolcu Sayısı: ",
                           font=("Helvetica", 10, "bold"), fill="white", tags="oneri_yazi2")

        canvas.create_text(303, 390, text=f"{en_kucuk}",
                           font=("Helvetica", 10, "bold"), fill="green", tags="oneri_yolcu")

        return tahmin

    def start_progress():
        max_value = 17000
        current_value = int(hesapla())

        percentage = (current_value / max_value) * 100
        progress["value"] = percentage

        canvas.itemconfig("progress_text", text=f"{int(percentage)}%")

        if percentage >= 90:
            progress.configure(bootstyle="danger")
        elif percentage >= 60:
            progress.configure(bootstyle="warning")
        else:
            progress.configure(bootstyle="success")

    def ongoru_sil(canvas):
        for item in canvas.find_all():
            tags = canvas.gettags(item)
            if "ongoru" in tags:
                canvas.delete(item)
            if "ongoru_yazi" in tags:
                canvas.delete(item)
            if "oneri_yazi1" in tags:
                canvas.delete(item)
            if "oneri_yazi2" in tags:
                canvas.delete(item)
            if "oneri_saat" in tags:
                canvas.delete(item)
            if "oneri_yolcu" in tags:
                canvas.delete(item)

    def hesapla2():
        ongoru_sil(canvas)
        hesapla()
        start_progress()

    def predict_passenger_count_LSTM(year, month, day, hour_sin, hour_cos, holiday, week_days, scaler_features,
                                     scaler_target, model):
        input_data = np.array([[year, month, day, hour_sin, hour_cos, holiday] + week_days])
        scaled_input = scaler_features.transform(input_data)
        sequence = np.expand_dims(scaled_input, axis=1)
        prediction = model.predict(sequence)
        return scaler_target.inverse_transform(prediction)

    ttk.Button(hesapla_frame, text="HESAPLA", style="danger", command=hesapla2).place(x=292, y=102)

    sayfayiGoster(yukleme_frame)
    root.after(3000, lambda: sayfayiGoster(anaSayfa_frame))

    root.yukleme_image = yukleme_image
    root.anaSayfa_image = anaSayfa_image
    root.m1bilgi_image = m1bilgi_image
    root.m2bilgi_image = m2bilgi_image
    root.m3bilgi_image = m3bilgi_image
    root.m4bilgi_image = m4bilgi_image
    root.m5bilgi_image = m5bilgi_image
    root.m6bilgi_image = m6bilgi_image
    root.m7bilgi_image = m7bilgi_image
    root.m8bilgi_image = m8bilgi_image
    root.m9bilgi_image = m9bilgi_image
    root.hakkinda_image = hakkinda_image
    root.hesaplamaSayfa_image = hesaplamaSayfa_image
    root.resizable(False, False)

    return root


if __name__ == "__main__":
    arayuz = arayuzOlustur()
    arayuz.mainloop()
