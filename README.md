# Klasifikasi Biji Kedelai - Streamlit Loader Model

Template aplikasi Streamlit untuk memuat model machine learning hasil training dan melakukan prediksi sederhana.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Catatan

- Gunakan file model `.pkl`, `.pickle`, atau `.joblib`.
- Jika model Anda memerlukan preprocessing, simpan preprocessing + model sebagai satu pipeline.
- Untuk input manual, masukkan fitur numerik sesuai urutan saat training.
