# %% [markdown]
# # Predictive Analytics

# %% [markdown]
# # Proyek Predictive Analytics: Prediksi Penyakit Jantung
# 
# - **Nama:** Bimoseno Kuma
# - **Email:** kuma24@student.ub.ac.id
# - **ID Dicoding:** kukuma
# 
# ## Domain Proyek
# Proyek ini berfokus pada domain kesehatan, khususnya dalam prediksi penyakit jantung. Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Kemampuan untuk memprediksi risiko penyakit jantung secara dini berdasarkan data klinis dapat membantu tenaga medis dalam memberikan intervensi yang lebih cepat dan efektif, sehingga berpotensi menyelamatkan nyawa.
# 
# ## Business Understanding
# Tujuan utama proyek ini adalah membangun model klasifikasi yang dapat memprediksi keberadaan penyakit jantung pada seorang pasien. Model ini diharapkan dapat menjadi alat bantu skrining awal bagi tenaga medis.
# 
# - **Problem Statement:** Bagaimana cara membangun model machine learning yang akurat dan andal untuk memprediksi penyakit jantung?
# - **Goals:** Mengembangkan model klasifikasi dengan F1-Score dan Recall setinggi mungkin untuk memaksimalkan identifikasi kasus positif.
# - **Solution:**
#     1.  Membangun dan membandingkan beberapa model klasifikasi (KNN dan Random Forest).
#     2.  Mengoptimalkan model terbaik menggunakan hyperparameter tuning.

# %% [markdown]
# ### Import Library
# Sel ini berisi semua library yang dibutuhkan untuk proyek, mulai dari manipulasi data, visualisasi, hingga pemodelan dan evaluasi.

# %%
# Import library 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

sns.set(style="whitegrid")

# %% [markdown]
# ## 1. Data Gathering
# 
# Tahap pertama adalah memuat dataset yang akan digunakan. Dataset yang digunakan adalah "Heart Disease UCI" yang bersumber dari Kaggle. Setelah memuat data, kita akan melakukan pemeriksaan awal untuk memahami struktur dan kondisi data.

# %%
df = pd.read_csv('heart_disease_uci.csv')

# %%
print("Lima baris pertama dari dataset:")
print(df.head())

# %%
print(f"Dimensi dataset: {df.shape[0]} baris dan {df.shape[1]} kolom")

# %%
print("Informasi Dasar DataFrame:")
df.info()

# %%
print("Jumlah Missing Values per Kolom:")
print(df.isnull().sum())

# %%
print("Statistik Deskriptif:")
print(df.describe())

# %% [markdown]
# ### Hasil dan Insight dari Data Gathering
# 
# Dari output pemeriksaan data di atas, kita mendapatkan beberapa wawasan dan temuan kunci yang akan memandu langkah-langkah selanjutnya:
# 
# 1.  **Dimensi Data:** Dataset terdiri dari **920 baris (sampel)** dan **16 kolom (fitur)**.
# 2.  **Tipe Data Campuran:** Output dari `.info()` menunjukkan adanya campuran tipe data. Terdapat 8 kolom bertipe `object` (contoh: `sex`, `cp`, `thal`) yang perlu diubah menjadi format numerik (proses *encoding*) agar bisa digunakan oleh model machine learning.
# 3.  **Adanya Missing Values:** Ini adalah temuan paling krusial. Output `.isnull().sum()` dengan jelas menunjukkan bahwa **terdapat banyak nilai yang hilang (NaN)** pada beberapa kolom. Kolom dengan jumlah *missing values* terbanyak adalah `ca` (611), `thal` (486), dan `slope` (309). Penanganan *missing values* ini akan menjadi prioritas utama di tahap *Data Preparation*.
# 4.  **Anomali pada Data Numerik:** Dari output `.describe()`, terlihat bahwa kolom `trestbps` (tekanan darah) dan `chol` (kolesterol) memiliki nilai minimum (`min`) 0.0. Secara medis, nilai ini tidak mungkin terjadi dan kemungkinan besar merupakan cara lain untuk merepresentasikan data yang hilang atau salah input. Ini adalah anomali data yang perlu diwaspadai.
# 
# **Kesimpulan:** Data mentah ini tidak bisa langsung digunakan untuk pemodelan. Diperlukan serangkaian proses pembersihan dan persiapan data (*Data Preparation*) untuk menangani masalah tipe data, nilai yang hilang, dan anomali yang telah teridentifikasi.

# %% [markdown]
# ## 2. Data Preparation
# 
# Berdasarkan temuan dari tahap sebelumnya, tahap Data Preparation ini akan mencakup semua proses untuk membersihkan dan menyiapkan data sebelum digunakan untuk analisis dan pemodelan.

# %% [markdown]
# ### 2.1. Transformasi dan Encoding Fitur
# 
# Langkah pertama dalam persiapan data adalah mengubah data ke format yang dapat diproses oleh model.
# 1.  **Transformasi Target:** Kolom `num` (target multi-kelas) diubah menjadi `target` biner (0=sehat, 1=sakit).
# 2.  **Encoding Kategorikal:** Fitur-fitur yang masih berupa teks (seperti 'sex', 'cp', dll.) diubah menjadi angka menggunakan teknik mapping.
# 3.  **Penghapusan Kolom:** Kolom `id` dan `dataset` dihapus karena tidak relevan untuk pemodelan. Kolom `num` yang asli juga dihapus karena telah digantikan oleh `target`

# %%
# Membuat Variabel Target Biner
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# %%
# Mengubah Kolom Boolean/Teks menjadi Numerik
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df['fbs'] = df['fbs'].map({True: 1, False: 0})
df['exang'] = df['exang'].map({True: 1, False: 0})

# %%
# Mengubah Fitur Kategorikal menjadi Numerik (Label Encoding)
cp_mapping = {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3}
restecg_mapping = {'normal': 0, 'lv hypertrophy': 2} 
slope_mapping = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
thal_mapping = {'normal': 3, 'fixed defect': 6, 'reversable defect': 7} 

df['cp'] = df['cp'].map(cp_mapping)
df['restecg'] = df['restecg'].map(restecg_mapping)
df['slope'] = df['slope'].map(slope_mapping)
df['thal'] = df['thal'].map(thal_mapping)

# %%
# Menghapus Kolom yang Tidak Diperlukan untuk Model
df_cleaned = df.drop(['id', 'dataset', 'num'], axis=1)
X = df_cleaned.drop('target', axis=1)
y = df_cleaned['target']


# %%
print("DataFrame Setelah Dibersihkan dan Diubah ke Format Numerik:")
print(df_cleaned.head())

# %%
print("\nJumlah missing values pada fitur (X) sebelum imputasi:")
print(X.isnull().sum())


# %% [markdown]
# ### Hasil dan Insight dari Transformasi
# 
# Dari output di atas, kita dapat menarik beberapa kesimpulan penting:
# 
# 1.  **Transformasi Berhasil:** Tabel `DataFrame Setelah Transformasi dan Encoding` menunjukkan bahwa kolom-kolom kategorikal (`sex`, `cp`, dll.) telah berhasil diubah menjadi format numerik, dan kolom `target` biner telah ditambahkan.
# 2.  **Missing Values Masih Ada:** Output dari `Jumlah missing values pada fitur (X) sebelum imputasi` mengungkapkan sebuah *insight* krusial. Meskipun kita sudah melakukan *mapping*, masih terdapat banyak sekali nilai kosong (NaN) pada beberapa fitur, seperti `ca` (611 nilai), `thal` (486 nilai), `slope` (309 nilai), serta pada fitur numerik seperti `trestbps` (59 nilai) dan `chol` (30 nilai).
# 3.  **Justifikasi Langkah Selanjutnya:** Temuan ini menegaskan bahwa proses `mapping` saja tidak cukup. Diperlukan langkah selanjutnya, yaitu **imputasi**, untuk menangani semua nilai kosong yang tersisa sebelum data dapat digunakan untuk pemodelan.

# %% [markdown]
# ### 2.2. Pembagian Dataset
# 
# Dataset dibagi menjadi data latih (80%) dan data uji (20%). Ini adalah langkah krusial yang dilakukan sebelum imputasi dan scaling untuk mencegah *data leakage*, yaitu kebocoran informasi dari data uji ke proses pelatihan model.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Ukuran data latih: {X_train.shape[0]} sampel")
print(f"Ukuran data uji: {X_test.shape[0]} sampel")

# %% [markdown]
# ### Hasil Pembagian Dataset
# -   **Data Latih:** Terdiri dari **736 sampel**, yang akan digunakan untuk melatih model machine learning kita. Semua proses seperti imputasi dan scaling akan di-*fit* pada data ini.
# -   **Data Uji:** Terdiri dari **184 sampel**, yang akan "disimpan" dan hanya digunakan pada tahap evaluasi akhir untuk mengukur seberapa baik performa model pada data baru.

# %% [markdown]
# ### 2.3. Penanganan Nilai Hilang dan Feature Scaling
# 
# Sekarang kita akan menangani semua nilai `NaN` yang tersisa di data latih dan data uji, kemudian melakukan scaling.
# 
# 1.  **Imputasi:** `SimpleImputer` digunakan untuk mengisi semua nilai `NaN` yang ada. Strategi `'median'` dipilih karena lebih robust terhadap outlier dibandingkan `'mean'`. Imputer di-*fit* hanya pada `X_train`.
# 2.  **Scaling:** `StandardScaler` digunakan untuk menyamakan skala semua fitur. Scaler juga di-*fit* hanya pada `X_train`.

# %%
# Menggunakan SimpleImputer untuk mengisi semua NaN yang tersisa
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Melakukan Scaling pada Fitur yang sudah diimputasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verifikasi akhir bahwa sudah tidak ada NaN
print("\nJumlah NaN di data latih setelah imputasi & scaling:", np.isnan(X_train_scaled).sum())
print("Jumlah NaN di data uji setelah imputasi & scaling:", np.isnan(X_test_scaled).sum())


# %%
print("Informasi Tipe Data Setelah Pembersihan:")
df_cleaned.info()

# %% [markdown]
# ### Hasil Penanganan Nilai Hilang dan Scaling
# 
# Output dari kode di atas memberikan konfirmasi penting:
# -   **Data Bersih dari Nilai Hilang:** Output `Jumlah NaN ...: 0` untuk data latih dan data uji membuktikan bahwa proses imputasi telah berhasil. Sekarang, tidak ada lagi nilai kosong di dalam data yang akan digunakan untuk melatih dan mengevaluasi model.
# -   **Data Siap untuk Pemodelan:** Dengan data yang telah bersih dari nilai `NaN` dan semua fitur yang telah diskalakan, `X_train_scaled` dan `X_test_scaled` kini siap sepenuhnya untuk digunakan dalam tahap *Exploratory Data Analysis* (EDA) lebih lanjut dan, yang terpenting, untuk melatih model machine learning.

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)
# 
# Setelah data bersih, kita melakukan analisis eksplorasi untuk mendapatkan wawasan.

# %% [markdown]
# ### 3.1. Distribusi Target Awal
# 
# Langkah pertama dalam EDA adalah memahami distribusi dari variabel target. Pada tahap ini, kita akan memvisualisasikan distribusi kolom `num`, yaitu variabel target asli sebelum kita melakukan transformasi menjadi biner.
# 
# Visualisasi ini bertujuan untuk melihat sebaran jumlah pasien untuk setiap kategori diagnosis, di mana:
# -   `0`: Pasien sehat (tidak ada penyakit jantung).
# -   `1-4`: Pasien dengan berbagai tingkat keparahan penyakit jantung.
# 
# Analisis ini akan memberikan pemahaman awal tentang keseimbangan kelas dalam data mentah dan menjadi dasar untuk keputusan transformasi pada tahap *Data Preparation*.

# %%
# Melihat distribusi variabel target
plt.figure(figsize=(6, 5))
sns.countplot(x='num', data=df, palette='pastel')
plt.title('Distribusi Pasien dengan dan Tanpa Penyakit Jantung')
plt.xlabel('Diagnosis (0 = Sehat, 1 = Sakit)')
plt.ylabel('Jumlah Pasien')
plt.show()

# %% [markdown]
# ### Hasil dan Insight dari Distribusi Target
# 
# Berdasarkan plot batang di atas, kita dapat menarik beberapa wawasan penting:
# 
# 1.  **Distribusi Kelas Tidak Merata:** Terlihat jelas bahwa jumlah pasien untuk setiap kategori tidak seimbang. Kelas `0` (pasien sehat) memiliki jumlah sampel terbanyak, yaitu lebih dari 400 pasien.
# 2.  **Tingkat Keparahan Penyakit:** Di antara pasien yang sakit, tingkat keparahan paling umum adalah level `1` (sekitar 260 pasien), sementara level `2`, `3`, dan `4` memiliki jumlah sampel yang jauh lebih sedikit.
# 3.  **Justifikasi Transformasi Biner:** Ketidakseimbangan yang signifikan antar kelas "sakit" (level 1-4) ini memberikan justifikasi yang kuat untuk menyederhanakan masalah. Menggabungkan semua level "sakit" (1, 2, 3, dan 4) menjadi satu kategori tunggal (target = 1) pada tahap *Data Preparation* adalah langkah yang logis untuk menciptakan masalah klasifikasi biner yang lebih seimbang dan dapat dikelola oleh model.

# %% [markdown]
# ### 3.2. Analisis Korelasi dan Distribusi Fitur
# 
# Pada tahap ini, kita akan melakukan dua jenis analisis untuk lebih memahami hubungan antar variabel dan karakteristik dari masing-masing fitur.
# 
# 1.  **Analisis Korelasi:** Kita akan membuat sebuah *heatmap* korelasi untuk memvisualisasikan hubungan linear antara setiap pasang fitur dalam dataset. Nilai korelasi berkisar dari -1 hingga 1. Nilai yang mendekati 1 menandakan hubungan positif yang kuat, nilai yang mendekati -1 menandakan hubungan negatif yang kuat, dan nilai yang mendekati 0 menandakan tidak adanya hubungan linear. Kita akan berfokus pada korelasi antara fitur-fitur independen dengan variabel `target`.
# 2.  **Analisis Distribusi Fitur Numerik:** Kita akan membuat histogram untuk beberapa fitur numerik kunci. Ini akan membantu kita memahami sebaran data, apakah simetris (normal), miring (skewed), atau memiliki karakteristik lain seperti adanya pencilan (outliers).

# %%
plt.figure(figsize=(14, 10))
correlation_matrix = df_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Antar Fitur', fontsize=18)
plt.show()

# %%
numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
df_cleaned[numerical_features].hist(bins=20, figsize=(15, 10), layout=(2, 3))
plt.suptitle('Distribusi Fitur Numerik', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% [markdown]
# ### Hasil dan Insight dari Analisis
# 
# Berdasarkan kedua visualisasi yang dihasilkan, kita mendapatkan beberapa wawasan penting:
# 
# #### Insight dari Heatmap Korelasi:
# -   **Korelasi Positif Kuat dengan Target:** Fitur `thalch` (detak jantung maks, +0.46), `cp` (jenis nyeri dada, +0.47), dan `slope` (+0.34) menunjukkan korelasi positif yang cukup kuat dengan variabel `target`. Ini mengindikasikan bahwa semakin tinggi nilai fitur-fitur ini, semakin tinggi pula kemungkinan seorang pasien didiagnosis menderita penyakit jantung.
# -   **Korelasi Negatif Kuat dengan Target:** Fitur `thal` (-0.50), `ca` (-0.46), `oldpeak` (-0.39), dan `exang` (-0.39) menunjukkan korelasi negatif yang kuat. Ini berarti semakin tinggi nilai pada fitur-fitur ini, semakin rendah kemungkinan pasien didiagnosis sakit.
# -   **Potensi Prediktor:** Fitur-fitur dengan korelasi kuat ini merupakan kandidat utama sebagai prediktor yang paling berpengaruh dalam model machine learning kita nanti.
# 
# #### Insight dari Distribusi Fitur Numerik:
# -   **Distribusi Normal:** Fitur `age`, `trestbps` (tekanan darah), dan `thalch` (detak jantung maks) menunjukkan distribusi yang mendekati kurva normal (lonceng), meskipun ada sedikit kemiringan.
# -   **Distribusi Miring (Skewed):** Fitur `oldpeak` sangat miring ke kanan (*right-skewed*), menandakan bahwa sebagian besar pasien memiliki nilai 0 atau mendekati 0 untuk metrik ini.
# -   **Anomali pada `chol`:** Terdapat sebuah anomali pada fitur `chol` (kolesterol), di mana ada lonjakan signifikan pada nilai 0. Nilai kolesterol 0 secara fisiologis tidak mungkin terjadi. Ini mengindikasikan bahwa nilai 0 kemungkinan besar digunakan untuk merepresentasikan data yang hilang pada sumber data asli. Meskipun kita telah menangani `NaN`, nilai "0" ini adalah bentuk lain dari data yang bermasalah yang dapat memengaruhi model. Namun, untuk saat ini kita akan melanjutkan analisis dengan data yang ada.

# %% [markdown]
# ## 4. Modeling
# 
# Tahap ini berfokus pada pembangunan dan perbandingan model machine learning.

# %% [markdown]
# ### 4.1. Model 1: K-Nearest Neighbors (KNN)
# 
# Model pertama yang dicoba adalah KNN dengan parameter default (`n_neighbors=5`).

# %%
# Inisialisasi dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Lakukan prediksi pada data uji
y_pred_knn = knn.predict(X_test_scaled)

# Tampilkan laporan klasifikasi untuk evaluasi
print("--- Hasil Evaluasi Model KNN Awal ---")
print(classification_report(y_test, y_pred_knn))

# %% [markdown]
# ### Hasil dan Insight dari Model KNN Awal
# 
# Dari `classification_report` yang dihasilkan, kita mendapatkan beberapa insight penting mengenai performa model KNN awal ini:
# 
# -   **Akurasi Keseluruhan:** Model mencapai akurasi sebesar **85%**, yang berarti model berhasil memprediksi dengan benar 85% dari total data uji.
# -   **Fokus pada Kelas 1 (Pasien Sakit):**
#     -   **Recall (0.89):** Ini adalah metrik yang sangat penting. Nilai 0.89 menunjukkan bahwa model berhasil mengidentifikasi **89%** dari semua pasien yang *sebenarnya* menderita penyakit jantung. Ini adalah hasil yang kuat karena model mampu meminimalkan jumlah kasus yang terlewat (*false negative*).
#     -   **Precision (0.84):** Dari semua pasien yang *diprediksi* sakit oleh model, **84%** di antaranya memang benar-benar sakit.
#     -   **F1-Score (0.87):** Skor ini menunjukkan adanya keseimbangan yang sangat baik antara *precision* dan *recall*, menandakan performa yang solid secara keseluruhan untuk kelas positif.

# %% [markdown]
# ### 4.2. Model 2: Random Forest
# 
# Model kedua yang kita evaluasi adalah Random Forest. Cara kerjanya adalah dengan membangun banyak *decision tree* (pohon keputusan) secara acak dari data latih, kemudian membuat prediksi berdasarkan suara mayoritas (voting) dari semua pohon tersebut.
# 
# Pendekatan ini membuat Random Forest cenderung lebih akurat dan lebih tahan terhadap *overfitting* dibandingkan dengan satu *decision tree* tunggal. Pada tahap ini, kita akan melatih model dengan `n_estimators=100`

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n--- Hasil Evaluasi Model Random Forest ---")
print(classification_report(y_test, y_pred_rf))

# %% [markdown]
# ### Hasil dan Insight dari Model Random Forest
# 
# Laporan klasifikasi untuk model Random Forest memberikan beberapa wawasan kunci:
# 
# -   **Akurasi Keseluruhan:** Model ini mencapai akurasi sebesar **86%**, sedikit lebih tinggi dari model KNN awal.
# -   **Fokus pada Kelas 1 (Pasien Sakit):**
#     -   **Recall (0.90):** Performa Recall sangat kuat, mencapai **90%**. Artinya, model ini berhasil mendeteksi 9 dari 10 pasien yang sebenarnya sakit di dalam data uji. Ini adalah pencapaian yang sangat baik untuk kasus medis.
#     -   **Precision (0.85):** Ketika model memprediksi seorang pasien "sakit", prediksinya benar sebanyak **85%**.
#     -   **F1-Score (0.88):** Skor ini, yang menyeimbangkan presisi dan recall, mencapai **0.88**, menunjukkan performa yang sangat solid dan seimbang.
# 
# -   **Perbandingan dengan KNN:** Jika dibandingkan dengan model KNN sebelumnya (F1-Score = 0.87), model Random Forest ini menunjukkan performa yang **sedikit lebih unggul** di hampir semua metrik utama. Berdasarkan hasil ini, Random Forest menjadi kandidat terkuat sebagai model terbaik.

# %% [markdown]
# ## 5. Hyperparameter Tuning
# 
# Kita akan melakukan eksperimen *hyperparameter tuning* pada model KNN. Tujuannya adalah untuk mencari kombinasi parameter terbaik bagi KNN dan melihat apakah performanya dapat dioptimalkan hingga melampaui Random Forest.
# 
# Proses ini akan menggunakan `GridSearchCV`, yang secara otomatis akan menguji semua kemungkinan kombinasi dari hyperparameter yang kita tentukan.

# %%
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1
)

print("Memulai proses Hyperparameter Tuning untuk KNN...")
knn_grid.fit(X_train_scaled, y_train)

print("\nParameter terbaik ditemukan:")
print(knn_grid.best_params_)

print("\nSkor F1 terbaik dari validasi silang:")
print(knn_grid.best_score_)


# %% [markdown]
# ### Hasil dan Insight dari Hyperparameter Tuning
# 
# -   **Proses Pencarian:** `GridSearchCV` telah melakukan proses yang komprehensif, dengan menguji **36 kandidat** kombinasi parameter sebanyak 5 kali (`5 folds`), sehingga total ada **180 proses pelatihan** model yang dilakukan untuk menemukan yang terbaik.
# -   **Parameter Terbaik Ditemukan:** Konfigurasi hyperparameter yang memberikan performa F1-score rata-rata tertinggi selama validasi silang adalah:
#     -   `metric`: 'manhattan'
#     -   `n_neighbors`: 13
#     -   `weights`: 'uniform'
#     Ini berarti, untuk dataset ini, model KNN bekerja paling optimal saat menggunakan 13 tetangga dengan metrik jarak Manhattan dan bobot yang seragam.
# -   **Skor Validasi Silang:** Skor F1 terbaik sebesar **0.831** adalah skor rata-rata yang didapat oleh konfigurasi terbaik di atas pada data latih.

# %% [markdown]
# ## 6. Evaluasi Model Final
# 
# Tahap terakhir adalah mengevaluasi performa model KNN yang telah di-tuning pada data uji. Model ini merupakan model final dari proyek ini.

# %%
best_knn_model = knn_grid.best_estimator_
y_pred_tuned = best_knn_model.predict(X_test_scaled)

print("\n--- Laporan Klasifikasi untuk Model KNN Final (Setelah Tuning) ---")
print(classification_report(y_test, y_pred_tuned))

# %% [markdown]
# ### Hasil dan Kesimpulan Akhir Proyek
# 
# Dari laporan klasifikasi di atas, model KNN yang telah di-tuning menghasilkan **F1-Score sebesar 0.85** untuk kelas positif (pasien sakit) pada data uji.
# 
# Sekarang mari kita rangkum performa dari semua model yang telah kita bangun:
# 
# 1.  **KNN (Awal / Baseline):**
#     -   F1-Score: 0.87
#     -   Recall: 0.89
# 2.  **Random Forest (Awal / Baseline):**
#     -   **F1-Score: 0.88**
#     -   **Recall: 0.90**
# 3.  **KNN (Setelah Tuning):**
#     -   F1-Score: 0.85
#     -   Recall: 0.85
# 
# #### Kesimpulan
# Setelah melalui seluruh tahapan, mulai dari persiapan data, pemodelan, hingga optimisasi, dapat disimpulkan bahwa **model Random Forest awal (dengan parameter default) adalah model terbaik**.
# 
# Meskipun proses *hyperparameter tuning* pada KNN telah dilakukan sebagai bagian dari solusi yang diusulkan, hasilnya (F1-Score 0.85) tidak berhasil melampaui performa model Random Forest (F1-Score 0.88). Eksperimen ini membuktikan model ensemble seperti Random Forest memberikan hasil yang lebih superior bahkan tanpa tuning.
# 
# Dengan Recall sebesar 90%, model Random Forest ini sangat efektif dalam mengidentifikasi mayoritas pasien yang benar-benar sakit, yang merupakan tujuan utamanya.


