import streamlit as st
import numpy as np
from PIL import Image

# Fungsi untuk memuat bobot model
def load_model(file_path):
    data = np.load(file_path)
    W1 = data['W1']
    b1 = data['b1']
    W2 = data['W2']
    b2 = data['b2']
    W3 = data['W3']
    b3 = data['b3']
    return W1, b1, W2, b2, W3, b3

# Fungsi forward pass
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = np.exp(Z3) / np.sum(np.exp(Z3), axis=1, keepdims=True)  # Softmax
    return Z1, A1, Z2, A2, Z3, A3

# Fungsi untuk memproses input pengguna
def process_user_input(umur, jenis_kelamin, tinggi_badan, max_umur, max_tinggi):
    jenis_kelamin_map = {'Laki-laki': 0, 'Perempuan': 1}
    jenis_kelamin_num = jenis_kelamin_map.get(jenis_kelamin)
    if jenis_kelamin_num is None:
        raise ValueError("Jenis Kelamin harus 'Laki-laki' atau 'Perempuan'")
    
    user_input = np.array([[umur, jenis_kelamin_num, tinggi_badan]])
    user_input[:, 0] = user_input[:, 0] / max_umur
    user_input[:, 2] = user_input[:, 2] / max_tinggi
    return user_input

# Fungsi prediksi
def predict_interactive(W1, b1, W2, b2, W3, b3, status_gizi_classes, max_umur, max_tinggi, umur, jenis_kelamin, tinggi_badan):
    try:
        user_input = process_user_input(umur, jenis_kelamin, tinggi_badan, max_umur, max_tinggi)
        _, _, _, _, _, A3 = forward_pass(user_input, W1, b1, W2, b2, W3, b3)
        predicted_class = status_gizi_classes[np.argmax(A3)]
        return predicted_class
    except ValueError as e:
        return f"Error: {e}. Silakan masukkan data yang valid."
    except Exception as e:
        return f"Terjadi kesalahan: {e}. Silakan coba lagi."

# Load model dan data
try:
    W1, b1, W2, b2, W3, b3 = load_model('model/stunted_model.npz')
    max_umur = 60
    max_tinggi = 127.9
    status_gizi_classes = np.array(['Normal', 'Stunted', 'Severely Stunted', 'Tinggi'])
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Konfigurasi halaman
st.set_page_config(page_title="Stunting Detection with Backpropagation", layout="wide", initial_sidebar_state="expanded")

# Header
st.markdown("# Prediksi Stunting pada Balita")
st.markdown("Aplikasi untuk memprediksi Stunting pada Balita dengan mudah!")

# Gambar utama dengan cropping tinggi
try:
    image = Image.open('data/baby.png')
    width, height = image.size
    crop_height = 100  # tinggi yang ingin ditampilkan

    # Hitung posisi atas dan bawah supaya crop dari tengah
    top = (height - crop_height) // 4
    bottom = top + crop_height

    cropped_image = image.crop((0, top, width, bottom))
    st.image(cropped_image, caption="Prediksi Status Stunting pada Balita", use_container_width=True)
except Exception as e:
    st.error(f"Gagal menampilkan gambar: {e}")

# Sidebar info
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.write("Aplikasi ini memakai model AI dengan metode backpropagation untuk memprediksi status stunting pada balita dari data umur, jenis kelamin, dan tinggi badan.")
    st.write("**Rentang Data:**")
    st.write("- Umur: 0-60 bulan")
    st.write("- Tinggi Badan: 0-127.9 cm")

# Input pengguna
col1, col2, col3 = st.columns(3)
with col1:
    umur = st.number_input("Umur (bulan)", min_value=0.0, max_value=60.0, step=1.0, format="%.1f")
with col2:
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
with col3:
    tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=0.0, max_value=127.9, step=0.1, format="%.1f")

# Tombol prediksi

if st.button("Prediksi Sekarang"):
    result = predict_interactive(W1, b1, W2, b2, W3, b3, status_gizi_classes, max_umur, max_tinggi, umur, jenis_kelamin, tinggi_badan)
    # Simulasi card dengan HTML & CSS
    st.markdown(
        f"""
        <div style="background-color:#f0f2f6; padding: 1.5em; border-radius: 12px; border: 1px solid #ddd; width: 100%; text-align: center; color: #333;">
            Prediksi Status: <br><b style="font-weight: bold; font-size: 1.5em;">{result}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.write("**Catatan Penting:** Pastikan data yang dimasukkan sesuai dengan rentang yang ditentukan.")

