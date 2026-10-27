import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Prediksi Modus Operandi Pencurian",
    page_icon="🚔",
    layout="centered"
)

@st.cache_resource
def load_files():
    model = joblib.load("model_curas_xgboost.pkl")
    prep = joblib.load("preprocessing_curas.pkl")
    return model, prep

try:
    model, prep = load_files()

    encoder = prep["encoder_fitur"]
    nominal_encoder = encoder.named_transformers_["nominal"]
    ordinal_encoder = encoder.named_transformers_["pemetaan"]

    kategori_nominal = dict(zip(
        prep["nominal_cols"],
        nominal_encoder.categories_
    ))

    kategori_ordinal = dict(zip(
        prep["ordinal_cols"],
        ordinal_encoder.categories_
    ))

    semua_kategori = {**kategori_nominal, **kategori_ordinal}

except Exception as e:
    st.error(f"Gagal memuat model atau preprocessing: {e}")
    st.stop()

st.title("🚔 Prediksi Modus Operandi Pencurian")

st.write(
    "Prediksi modus operandi berdasarkan karakteristik "
    "korban dan peristiwa pencurian."
)

st.info("Model: XGBoost | Training: setelah SMOTEN")

kolom_korban = [
    "JENIS KELAMIN KORBAN",
    "USIA KORBAN",
    "PEKERJAAN KORBAN",
    "DAERAH TINGGAL KORBAN"
]

kolom_kejadian = [
    "TEMPAT KEJADIAN PENCURIAN",
    "DAERAH PERISTIWA",
    "WAKTU KEJADIAN",
    "HARI KEJADIAN",
    "PELAKU MENGGUNAKAN ALAT",
    "SASARAN KEJAHATAN",
    "JUMLAH KERUGIAN"
]

input_data = {}

with st.form("form_prediksi"):
    st.subheader("1. Informasi Korban")

    for kolom in kolom_korban:
        input_data[kolom] = st.selectbox(
            kolom.title(),
            options=list(semua_kategori[kolom])
        )

    st.subheader("2. Informasi Kejadian")

    for kolom in kolom_kejadian:
        input_data[kolom] = st.selectbox(
            kolom.title(),
            options=list(semua_kategori[kolom])
        )

    tombol_prediksi = st.form_submit_button(
        "🔍 Prediksi Modus Operandi",
        use_container_width=True
    )

if tombol_prediksi:
    try:
        data_baru = pd.DataFrame(
            [input_data],
            columns=prep["feature_columns"]
        )

        for kolom in data_baru.columns:
            data_baru[kolom] = (
                data_baru[kolom]
                .astype(str)
                .str.strip()
                .str.upper()
            )

        hasil_encoding = encoder.transform(data_baru)

        X_encoded = pd.DataFrame(
            hasil_encoding,
            columns=prep["encoded_feature_columns"]
        )

        prob_awal = model.predict_proba(X_encoded)[0]

        kelas_target = prep["encoder_target"].classes_
        probabilitas = np.zeros(len(kelas_target))
        probabilitas[np.asarray(model.classes_, dtype=int)] = prob_awal

        kode_prediksi = int(np.argmax(probabilitas))
        hasil_prediksi = kelas_target[kode_prediksi]

        st.divider()
        st.subheader("3. Hasil Prediksi")

        st.success(f"Prediksi Modus Operandi: {hasil_prediksi}")
        st.metric(
            "Probabilitas Prediksi",
            f"{probabilitas[kode_prediksi] * 100:.2f}%"
        )

        hasil = pd.DataFrame({
            "Modus Operandi": kelas_target,
            "Probabilitas (%)": probabilitas * 100
        }).sort_values("Probabilitas (%)", ascending=False)

        st.subheader("Probabilitas Setiap Kelas MO")
        st.dataframe(hasil, use_container_width=True, hide_index=True)
        st.bar_chart(
            hasil.set_index("Modus Operandi")["Probabilitas (%)"]
        )

        st.caption(
            "Hasil merupakan prediksi berdasarkan data pelatihan, "
            "bukan kepastian kejadian."
        )

    except Exception as e:
        st.error(f"Prediksi gagal: {e}")
