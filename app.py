
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Prediksi Modus Operandi Pencurian",
    page_icon="🚔",
    layout="centered"
)

# ===========================
# LOAD MODEL & PREPROCESSING
# ===========================

@st.cache_resource
def load_files():
    prep = joblib.load("preprocessing_curas.pkl")

    if prep["nama_model"] == "CNN1D":
        from tensorflow.keras.models import load_model
        model = load_model("model_curas.keras", compile=False)
    else:
        model = joblib.load("model_curas.pkl")

    return model, prep


try:
    model, prep = load_files()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# ===========================
# HEADER
# ===========================

st.title("🚔 Prediksi Modus Operandi Pencurian")

st.write(
    "Aplikasi Machine Learning untuk memprediksi "
    "Modus Operandi (MO) pencurian berdasarkan "
    "karakteristik korban dan peristiwa kejahatan."
)

st.info(
    f"Model: {prep['nama_model']} | "
    f"Training: {prep['skenario']}"
)

st.divider()


# ===========================
# PILIHAN KATEGORI
# ===========================

encoder = prep["encoder_fitur"]

nominal_encoder = encoder.named_transformers_["nominal"]
ordinal_encoder = encoder.named_transformers_["pemetaan"]

kategori_nominal = {
    kolom: list(kategori)
    for kolom, kategori in zip(
        prep["kolom_nominal"],
        nominal_encoder.categories_
    )
}

kategori_pemetaan = {
    kolom: list(kategori)
    for kolom, kategori in zip(
        prep["kolom_pemetaan"],
        ordinal_encoder.categories_
    )
}

semua_kategori = {
    **kategori_nominal,
    **kategori_pemetaan
}


# ===========================
# FORM INPUT
# ===========================

st.subheader("1. Informasi Korban")

input_data = {}

with st.form("form_prediksi"):

    kolom_korban = [
        "JENIS KELAMIN KORBAN",
        "USIA KORBAN",
        "PEKERJAAN KORBAN",
        "DAERAH TINGGAL KORBAN"
    ]

    for kolom in kolom_korban:
        input_data[kolom] = st.selectbox(
            kolom.title(),
            semua_kategori[kolom]
        )

    st.subheader("2. Informasi Kejadian")

    kolom_kejadian = [
        "TEMPAT KEJADIAN PENCURIAN",
        "DAERAH PERISTIWA",
        "WAKTU KEJADIAN",
        "HARI KEJADIAN",
        "PELAKU MENGGUNAKAN ALAT",
        "SASARAN KEJAHATAN",
        "JUMLAH KERUGIAN"
    ]

    for kolom in kolom_kejadian:
        input_data[kolom] = st.selectbox(
            kolom.title(),
            semua_kategori[kolom]
        )

    tombol_prediksi = st.form_submit_button(
        "🔍 Prediksi Modus Operandi",
        use_container_width=True
    )


# ===========================
# PROSES PREDIKSI
# ===========================

if tombol_prediksi:

    # Data input dengan urutan fitur sesuai training
    data_baru = pd.DataFrame(
        [input_data],
        columns=prep["fitur_input"]
    )

    for kolom in data_baru.columns:
        data_baru[kolom] = (
            data_baru[kolom]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Encoding menggunakan encoder training
    X_encoded = prep["encoder_fitur"].transform(
        data_baru
    )

    X_encoded = pd.DataFrame(
        X_encoded,
        columns=prep["nama_fitur"]
    )

    # ===========================
    # PREDIKSI MODEL
    # ===========================

    if prep["nama_model"] == "CNN1D":

        X_cnn = np.asarray(
            X_encoded,
            dtype=np.float32
        )[..., np.newaxis]

        probabilitas = model.predict(
            X_cnn,
            verbose=0
        )[0]

    else:

        prob_awal = model.predict_proba(
            X_encoded
        )[0]

        # Urutkan probabilitas berdasarkan kode target
        jumlah_kelas = len(
            prep["encoder_target"].classes_
        )

        probabilitas = np.zeros(jumlah_kelas)

        probabilitas[
            np.asarray(model.classes_, dtype=int)
        ] = prob_awal

    kode_prediksi = int(
        np.argmax(probabilitas)
    )

    hasil_prediksi = (
        prep["encoder_target"]
        .inverse_transform([kode_prediksi])[0]
    )

    confidence = float(
        probabilitas[kode_prediksi] * 100
    )

    # ===========================
    # TAMPILKAN HASIL
    # ===========================

    st.divider()

    st.subheader("3. Hasil Prediksi")

    st.success(
        f"Prediksi Modus Operandi: {hasil_prediksi}"
    )

    st.metric(
        "Probabilitas Prediksi",
        f"{confidence:.2f}%"
    )

    st.subheader("Probabilitas Setiap Kelas MO")

    hasil_probabilitas = pd.DataFrame({
        "Modus Operandi": (
            prep["encoder_target"].classes_
        ),
        "Probabilitas (%)": (
            probabilitas * 100
        )
    })

    hasil_probabilitas = (
        hasil_probabilitas
        .sort_values(
            "Probabilitas (%)",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        hasil_probabilitas,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        hasil_probabilitas.set_index(
            "Modus Operandi"
        )["Probabilitas (%)"]
    )

    st.caption(
        "Hasil merupakan prediksi statistik berdasarkan "
        "data pelatihan, bukan kepastian kejadian."
    )
