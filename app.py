import streamlit as st
import pandas as pd
import joblib
import numpy as np
import logging

# Setup logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load model, encoder, dan data produk
try:
    model = joblib.load('iphone_quantity_model.pkl')
    encoder = joblib.load('product_id_encoder.pkl')
    product_data = joblib.load('product_data.pkl')
except FileNotFoundError as e:
    st.error(f"Error: File tidak ditemukan - {e}")
    logging.error(f"File not found: {e}")
    st.stop()

# Dictionary untuk lookup produk
product_dict = {f"{p['product_name']} ({p['storage']}, {p['color']})": p for p in product_data}

# Judul aplikasi
st.title("Prediksi Jumlah Pembelian iPhone")

# Deskripsi
st.write("Masukkan detail transaksi untuk memvalidasi harga dan memprediksi jumlah pembelian.")

# Input pengguna
st.subheader("Input Transaksi")
product_selection = st.selectbox("Pilih Produk", options=list(product_dict.keys()))

# Ambil detail produk sekali
selected_product = product_dict[product_selection]
product_id = selected_product['product_id']
reference_price = selected_product['price']

# Input harga (otomatis atau manual)
use_manual_price = st.checkbox("Masukkan Harga Manual")
if use_manual_price:
    unit_price = st.number_input(
        "Harga Satuan (Rp)",
        min_value=1000000,
        max_value=30000000,
        step=100000,
        value=int(reference_price)  # Pastikan integer
    )
else:
    unit_price = reference_price
st.write(f"**Harga Satuan yang Digunakan**: Rp {unit_price:,.0f}")

# Input diskon
discount = st.selectbox("Pilih Diskon (%)", options=[0, 5, 10, 15], index=0)

# Validasi diskon
if not isinstance(discount, (int, float)) or discount not in [0, 5, 10, 15]:
    st.error("Error: Diskon harus salah satu dari 0, 5, 10, atau 15 persen.")
    logging.error(f"Invalid discount value: {discount}")
    st.stop()

# Tombol Validasi Harga
if st.button("Validasi Harga", type="primary", use_container_width=True):
    try:
        # Hitung harga setelah diskon
        price_after_discount = unit_price * (1 - discount / 100)
        st.write(f"**Harga Setelah Diskon**: Rp {price_after_discount:,.0f}")

        # Validasi harga
        if abs(unit_price - reference_price) > 1000000:
            st.error(
                f"⚠️ Harga unit tidak sesuai referensi untuk {product_selection} "
                f"(Referensi: Rp {reference_price:,.0f})"
            )
        else:
            st.success(
                f"✅ Harga unit sesuai referensi untuk {product_selection} "
                f"(Referensi: Rp {reference_price:,.0f})"
            )
    except Exception as e:
        st.error("Error: Gagal menghitung harga setelah diskon. Periksa input.")
        logging.error(f"Price calculation error: {e}")
        st.stop()

# Tombol Prediksi
if st.button("Prediksi", type="secondary", use_container_width=True):
    # Siapkan data input untuk prediksi
    input_data = {
        'unit_price': [unit_price],
        'discount': [discount]
    }
    # One-hot encoding product_id
    try:
        product_id_array = np.array([[product_id]])
        product_id_encoded = encoder.transform(product_id_array)
        product_id_df = pd.DataFrame(
            product_id_encoded,
            columns=[f'product_id_{int(i)}' for i in encoder.categories_[0]]
        )
    except Exception as e:
        st.error("Error: Gagal melakukan encoding product_id.")
        logging.error(f"Encoding error: {e}")
        st.stop()

    # Gabungkan fitur
    input_df = pd.DataFrame(input_data)
    input_df = pd.concat([input_df, product_id_df], axis=1)

    # Pastikan kolom sesuai dengan model
    try:
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    except AttributeError:
        st.error("Error: Model tidak memiliki atribut feature_names_in_. Periksa kompatibilitas model.")
        logging.error("Model missing feature_names_in_")
        st.stop()

    # Prediksi
    try:
        predicted_quantity = model.predict(input_df)[0]
        st.success(f"📦 **Prediksi Jumlah Pembelian**: {predicted_quantity:.2f} unit")

        # Catatan tentang prediksi
        if predicted_quantity < 1 or predicted_quantity > 4:
            st.warning("Catatan: Prediksi mungkin tidak realistis karena jumlah pembelian harus antara 1-4 unit.")
    except Exception as e:
        st.error("Error: Gagal melakukan prediksi.")
        logging.error(f"Prediction error: {e}")
        st.stop()
