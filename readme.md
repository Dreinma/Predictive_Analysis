# Proyek Predictive Analytics: Prediksi Penyakit Jantung

Proyek ini merupakan bagian dari submission kelas "Belajar Pengembangan Machine Learning" oleh Dicoding.

## Ringkasan Proyek

Proyek ini bertujuan untuk membangun model klasifikasi guna memprediksi keberadaan penyakit jantung pada pasien berdasarkan 13 atribut medis. Tujuan utamanya adalah mengembangkan model yang dapat diandalkan sebagai alat bantu skrining awal bagi tenaga medis.

Setelah melalui tahapan persiapan data, perbandingan beberapa model, dan eksperimen optimisasi, model akhir yang dipilih adalah **Random Forest**. Model ini menunjukkan performa terbaik dengan **F1-Score 0.88** dan **Recall 0.90** pada data uji, yang menandakan keseimbangan yang baik antara presisi dan kemampuan untuk mendeteksi kasus positif.

## Cara Menjalankan Proyek

Untuk menjalankan proyek ini dan mereproduksi hasilnya, ikuti langkah-langkah berikut:

1.  **Klon Repositori**
    ```bash
    git clone https://github.com/Dreinma/Predictive_Analysis.git
    cd https://github.com/Dreinma/Predictive_Analysis.git
    ```

2.  **Instalasi Dependensi**
    Pastikan Anda memiliki Python dan pip. Jalankan perintah berikut untuk menginstal semua library yang diperlukan:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3.  **Jalankan Notebook**
    Buka `Notebook.ipynb` menggunakan Jupyter Notebook, JupyterLab, atau Visual Studio Code. Untuk mereproduksi semua hasil, jalankan semua sel secara berurutan dari atas ke bawah.

## Sumber Data

Dataset yang digunakan adalah "Heart Disease UCI" yang tersedia di Kaggle:
[https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)