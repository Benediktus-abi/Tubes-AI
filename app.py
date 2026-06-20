import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ==========================================
# 1. LOAD MODEL & ALAT PREPROCESSING
# ==========================================
# Pastikan 4 file .pkl ini berada di satu folder yang sama dengan app.py
try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('encoder_dict.pkl', 'rb') as file:
        encoder_dict = pickle.load(file)
    with open('fitur_model.pkl', 'rb') as file:
        fitur_model = pickle.load(file)
except FileNotFoundError:
    st.error("File .pkl tidak ditemukan! Pastikan best_model, scaler, encoder_dict, dan fitur_model sudah ada di folder.")

# ==========================================
# 2. TAMPILAN UTAMA & METRIK AKURASI
# ==========================================
st.title("🎓 Prediksi IPK Mahasiswa")
st.write("Aplikasi ini memprediksi nilai CGPA (IPK) berdasarkan kebiasaan dan gaya hidup mahasiswa.")

# Wajib: Mencantumkan nilai akurasi di halaman website sesuai instruksi tugas
st.info("📊 Performa Model (Linear Regression): R² = 0.85 | Akurasi Toleransi (±0.5) = 87.50%")

# ==========================================
# 3. FORM INPUT PARAMETER (Min. 4 Input)
# ==========================================
st.subheader("Masukkan Data Mahasiswa")

# Membagi tampilan menjadi 2 kolom agar rapi
col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Jam Belajar per Hari", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    sleep_hours = st.number_input("Jam Tidur per Hari", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
    screen_time = st.number_input("Screen Time per Hari (Jam)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    attendance = st.number_input("Persentase Kehadiran (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

with col2:
    stress_level = st.slider("Tingkat Stres (1-10)", min_value=1, max_value=10, value=5)
    internal_marks = st.number_input("Nilai Internal (bebas)", min_value=0.0, max_value=200.0, value=75.0, step=1.0)
    diet_type = st.selectbox("Pola Makan (Diet Type)", ["Veg", "Non-Veg"])
    residence = st.selectbox("Tempat Tinggal", ["Day Scholar", "Hosteller"])

# Input default (tersembunyi) agar jumlah fitur pas dengan model saat ditraining
# Misalnya fitur 'Age', 'Gym_Hours_per_Week', dan 'Branch' kita beri nilai rata-rata/umum 
# agar pengguna tidak pusing mengisi terlalu banyak form
age_default = 21
gym_default = 2.0
branch_default = "CSE" 

# ==========================================
# 4. PEMROSESAN INPUT & PREDIKSI
# ==========================================
if st.button("Hitung Prediksi IPK"):
    # a. Kalkulasi fitur tambahan hasil feature engineering (Phase 1 Cell 6)
    study_screen_ratio = study_hours / (screen_time + 1e-5)
    
    # b. Menyusun data input mentah sesuai teks
    input_data = {
        'Age': age_default,
        'Study_Hours_per_Day': study_hours,
        'Sleep_Hours': sleep_hours,
        'Screen_Time_Hours': screen_time,
        'Gym_Hours_per_Week': gym_default,
        'Diet_Type': diet_type,
        'Attendance_Percentage': attendance,
        'Stress_Level_1_to_10': stress_level,
        'Residence': residence,
        'Internal_Marks': internal_marks,
        'Study_to_Screen_Ratio': study_screen_ratio,
        'Branch': branch_default
    }
    
    # Ubah menjadi DataFrame (1 baris)
    df_input = pd.DataFrame([input_data])
    
    # c. Encoding data kategorikal pakai dictionary yang sudah disimpan
    for col in ['Diet_Type', 'Residence']:
        if col in encoder_dict:
            df_input[col] = encoder_dict[col].transform(df_input[col])
            
    # One-Hot Encoding untuk 'Branch' seperti saat training
    df_input = pd.get_dummies(df_input, columns=['Branch'], drop_first=True)
    
    # d. Memastikan kolom sesuai dengan fitur_model (menambah kolom 0 jika ada yang kurang karena One-Hot Encoding)
    for col in fitur_model:
        if col not in df_input.columns:
            df_input[col] = 0
            
    # Mengurutkan kolom agar persis dengan urutan saat training model
    df_input = df_input[fitur_model]
    
    # e. Standarisasi (Scaling) hanya pada kolom numerik asli
    kolom_numerik = ['Age', 'Study_Hours_per_Day', 'Sleep_Hours', 'Screen_Time_Hours', 
                     'Gym_Hours_per_Week', 'Attendance_Percentage', 'Stress_Level_1_to_10', 
                     'Internal_Marks', 'Study_to_Screen_Ratio']
    
    df_input[kolom_numerik] = scaler.transform(df_input[kolom_numerik])
    
    # f. Eksekusi Prediksi
    hasil_prediksi = model.predict(df_input)[0]
    
    # Memastikan nilai prediksi mentok di 10.0
    hasil_prediksi = min(hasil_prediksi, 10.0)
    
    # ==========================================
    # 5. MENAMPILKAN HASIL
    # ==========================================
    st.success(f"📈 Prediksi CGPA (IPK): **{hasil_prediksi:.2f}**")
