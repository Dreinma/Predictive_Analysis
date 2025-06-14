# Laporan Proyek Machine Learning - Prediksi Penyakit Jantung

## Domain Proyek
Penyakit kardiovaskular atau *Cardiovascular Diseases* (CVDs) merupakan penyebab utama kematian secara global. Menurut Organisasi Kesehatan Dunia (WHO), diperkirakan 17,9 juta orang meninggal akibat CVDs pada tahun 2019, yang mewakili 32% dari semua kematian global. Dari kematian tersebut, 85% disebabkan oleh serangan jantung dan stroke. Tingginya angka kematian ini menunjukkan betapa krusialnya deteksi dini dan diagnosis yang akurat untuk pencegahan dan penanganan yang efektif.

Masalah ini harus segera diselesaikan karena diagnosis dini dapat secara signifikan meningkatkan peluang keberhasilan pengobatan dan menurunkan tingkat mortalitas. Namun, proses diagnosis konvensional sering kali memerlukan serangkaian tes yang memakan waktu dan biaya, serta bergantung pada keahlian dan pengalaman dokter spesialis yang jumlahnya terbatas.

Oleh karena itu, pemanfaatan teknologi *machine learning* dapat menjadi solusi yang menjanjikan. Dengan menganalisis data klinis pasien yang umum dikumpulkan saat pemeriksaan rutin, model *machine learning* dapat dikembangkan untuk mengidentifikasi pola dan memprediksi kemungkinan seseorang menderita penyakit jantung. Proyek ini bertujuan untuk membangun sebuah model prediktif yang dapat membantu tenaga medis sebagai alat bantu skrining awal.

**Referensi:**
- World Health Organization. (2021, Juni 11). *Cardiovascular diseases (CVDs)*. Diakses dari https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

## Business Understanding

### Problem Statements
1.  Proses diagnosis penyakit jantung secara manual bersifat kompleks dan memakan waktu, sehingga menghambat deteksi dini pada skala besar.
2.  Bagaimana cara membangun sebuah model klasifikasi yang akurat dan andal untuk memprediksi keberadaan penyakit jantung pada pasien berdasarkan data klinis rutin?
3.  Fitur atau atribut medis manakah yang paling berpengaruh dalam memprediksi risiko penyakit jantung untuk memberikan wawasan tambahan bagi para profesional medis?

### Goals
1.  Mengembangkan sistem prediktif otomatis yang dapat berfungsi sebagai alat bantu skrining awal.
2.  Membuat model *machine learning* dengan F1-Score dan Recall setinggi mungkin untuk memaksimalkan identifikasi pasien yang benar-benar sakit.
3.  Menganalisis dan mengidentifikasi fitur-fitur paling signifikan yang digunakan model dalam membuat prediksi.

### Solution Statements
1.  **Pengembangan dan Perbandingan Model:** Membangun dan mengevaluasi dua model klasifikasi, yaitu K-Nearest Neighbors (KNN) dan Random Forest, untuk memilih model dengan performa F1-Score terbaik sebagai dasar.
2.  **Optimisasi Model:** Melakukan eksperimen optimisasi menggunakan *hyperparameter tuning* dengan `GridSearchCV` untuk melihat potensi peningkatan performa lebih lanjut.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Heart Disease UCI". Proses pemuatan data menunjukkan bahwa dataset mentah terdiri dari **920 baris data** dan **16 kolom**.

**Tautan Dataset:** [Heart Disease UCI di Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

### Kondisi Awal Data
Pemeriksaan awal menggunakan `.info()` dan `.isnull().sum()` mengungkapkan beberapa kondisi penting yang perlu ditangani:
* **Tipe Data Campuran**: Terdapat 8 kolom dengan tipe data `object` (contoh: `sex`, `cp`, `thal`) yang perlu diubah menjadi format numerik.
* **Nilai Hilang (Missing Values)**: Ditemukan adanya nilai kosong (NaN) dalam jumlah signifikan pada banyak kolom, termasuk `ca` (611), `thal` (486), `slope` (309), `trestbps` (59), dan `chol` (30).
* **Anomali Data**: Pada ringkasan statistik, kolom `trestbps` dan `chol` memiliki nilai minimum 0, yang secara medis tidak mungkin dan mengindikasikan adanya data yang tidak valid.

Temuan ini menegaskan bahwa data mentah tidak dapat langsung digunakan dan memerlukan tahap **Data Preparation** yang komprehensif.

### Deskripsi Variabel
Berikut adalah penjelasan untuk setiap variabel (fitur) pada dataset:
- **id**: ID unik pasien.
- **age**: Usia pasien dalam tahun.
- **sex**: Jenis kelamin pasien (Male/Female).
- **dataset**: Sumber dataset (misalnya, Cleveland, Hungarian).
- **cp** (chest pain type): Tipe nyeri dada.
- **trestbps**: Tekanan darah pasien saat istirahat (mm Hg).
- **chol**: Kadar kolesterol serum (mg/dl).
- **fbs** (fasting blood sugar): Kadar gula darah puasa > 120 mg/dl (True/False).
- **restecg**: Hasil elektrokardiogram saat istirahat.
- **thalch**: Detak jantung maksimum yang tercatat.
- **exang** (exercise induced angina): Angina akibat olahraga (True/False).
- **oldpeak**: Depresi segmen ST akibat olahraga.
- **slope**: Kemiringan puncak segmen ST saat olahraga.
- **ca**: Jumlah pembuluh darah utama yang terlihat.
- **thal**: Status thalasemia.
- **num**: Variabel target asli (0: tidak ada penyakit, 1-4: berbagai tingkat penyakit).

## Data Preparation

Berdasarkan temuan dari tahap *Data Understanding*, dilakukan serangkaian proses persiapan data secara berurutan:

1.  **Transformasi dan Encoding**:
    * **Variabel Target**: Kolom `num` diubah menjadi target biner `target` (0 = sehat, 1 = sakit).
    * **Fitur Kategorikal**: Kolom berformat teks dan boolean diubah menjadi numerik menggunakan teknik pemetaan (`mapping`).
2.  **Penghapusan Kolom**: Kolom yang tidak relevan (`id`, `dataset`, `num`) dihapus.
3.  **Pembagian Dataset**: Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`. Parameter `stratify=y` digunakan untuk menjaga proporsi kelas yang seimbang di kedua set.
4.  **Penanganan Nilai Hilang (Imputation)**: Setelah pembagian data, `SimpleImputer` dengan strategi `'median'` digunakan untuk mengisi semua nilai `NaN` yang tersisa. Median dipilih karena lebih robust terhadap outlier. Imputer ini di-*fit* hanya pada data latih untuk mencegah *data leakage*.
5.  **Feature Scaling**: Terakhir, `StandardScaler` diterapkan untuk menyamakan skala semua fitur. Ini krusial untuk algoritma berbasis jarak seperti KNN. Scaler juga di-*fit* hanya pada data latih.

## Exploratory Data Analysis (EDA)

Analisis eksplorasi dilakukan pada data yang telah dibersihkan untuk mendapatkan wawasan lebih dalam.

#### Distribusi Target
Visualisasi distribusi variabel target (`target`) menunjukkan dataset yang cukup seimbang, terdiri dari **526 sampel (57%)** untuk kelas "sakit" dan **394 sampel (43%)** untuk kelas "sehat".

![Distribusi Target](https://github.com/user-attachments/assets/214cd1c7-ed7f-4237-8a04-4e01f4554998)

#### Analisis Korelasi Fitur
*Heatmap* korelasi menunjukkan bahwa fitur `cp`, `thalch`, dan `slope` memiliki korelasi positif yang kuat dengan target, sementara `thal`, `ca`, `oldpeak`, dan `exang` memiliki korelasi negatif yang kuat.

![Heatmap Fitur](https://github.com/user-attachments/assets/98c25945-ec49-4335-a701-307dacc6c197)


#### Distribusi Fitur Numerik
Histogram menunjukkan bahwa fitur `age` dan `trestbps` terdistribusi mendekati normal, sedangkan `oldpeak` sangat miring ke kanan (*right-skewed*).

![Distribusi Fitur Numerik](https://github.com/user-attachments/assets/aa207fe0-8fb8-4987-ae9d-24feedf1e731)

## Modeling

Tahap ini berfokus pada pelatihan, perbandingan, dan optimisasi model machine learning.

### K-Nearest Neighbors (KNN)

**Cara Kerja Algoritma**
K-Nearest Neighbors (KNN) adalah algoritma *supervised learning* yang bekerja berdasarkan prinsip "kesamaan". Untuk mengklasifikasikan sebuah data baru, algoritma ini akan mencari 'K' jumlah tetangga terdekat dari data tersebut di dalam data latih, berdasarkan perhitungan jarak (misalnya, Jarak Euclidean atau Manhattan). Kelas dari data baru kemudian ditentukan oleh suara mayoritas (voting) dari kelas-kelas para tetangga terdekatnya.

**Parameter dan Pelatihan Model Awal**
Model KNN awal dilatih menggunakan parameter default dari library scikit-learn, dengan parameter utama yang ditetapkan secara eksplisit adalah:
-   `n_neighbors=5`

**Kelebihan dan Kekurangan**
-   **Kelebihan**:
    -   Sederhana untuk dipahami dan diimplementasikan.
    -   Tidak memerlukan asumsi tentang distribusi data (non-parametrik).
    -   Efektif untuk data yang tidak terlalu besar.
-   **Kekurangan**:
    -   Sangat sensitif terhadap skala fitur, sehingga memerlukan *feature scaling*.
    -   Biaya komputasi menjadi tinggi saat proses prediksi karena perlu menghitung jarak ke semua data latih.
    -   Performa dapat menurun pada dataset dengan dimensi (jumlah fitur) yang sangat tinggi.

Hasil evaluasi awal pada data uji menunjukkan model ini menghasilkan **F1-Score 0.87** untuk kelas positif.

### Random Forest

**Cara Kerja Algoritma**
Random Forest adalah algoritma *ensemble learning* yang terdiri dari banyak *decision tree* (pohon keputusan). Prosesnya dimulai dengan membuat sejumlah besar pohon keputusan secara acak, di mana setiap pohon dilatih pada sampel data yang sedikit berbeda (teknik *bootstrap aggregating* atau *bagging*). Untuk membuat prediksi klasifikasi, setiap pohon di dalam "hutan" akan memberikan suaranya (voting), dan kelas dengan suara terbanyak akan menjadi hasil prediksi akhir dari model.

**Parameter dan Pelatihan Model Awal**
Model Random Forest awal dilatih dengan parameter berikut:
-   `n_estimators=100` (jumlah pohon keputusan yang dibangun).
-   `random_state=42` (untuk memastikan hasil yang dapat direproduksi).

**Kelebihan dan Kekurangan**
-   **Kelebihan**:
    -   Memiliki akurasi yang tinggi dan cenderung sangat robust terhadap *overfitting*.
    -   Dapat menangani data dalam jumlah besar dengan fitur yang banyak.
    -   Mampu memberikan peringkat pentingnya setiap fitur (*feature importance*).
-   **Kekurangan**:
    -   Cenderung menjadi "black box", artinya proses pengambilan keputusannya lebih sulit diinterpretasikan dibandingkan satu decision tree.
    -   Membutuhkan lebih banyak sumber daya komputasi dan waktu untuk melatih model karena membangun banyak pohon.

Hasil evaluasi awal pada data uji menunjukkan model ini menghasilkan **F1-Score 0.88** untuk kelas positif.

### Pemilihan Model dan Eksperimen Optimisasi

**Pemilihan Model Terbaik**
Berdasarkan perbandingan F1-Score awal (Random Forest: 0.88 vs. KNN: 0.87), **Random Forest dipilih sebagai model terbaik** karena menunjukkan performa yang sedikit lebih unggul.

**Eksperimen Hyperparameter Tuning**
Sebagai bagian dari *solution statement*, sebuah eksperimen optimisasi tetap dilakukan pada model KNN menggunakan `GridSearchCV` untuk melihat apakah performanya dapat ditingkatkan hingga melampaui Random Forest. 
-   **Parameter Terbaik Ditemukan**: `{'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'uniform'}`.
-   **Hasil**: Model KNN yang telah di-tuning dievaluasi dan menghasilkan **F1-Score 0.85** pada data uji.

Hasil eksperimen ini mengkonfirmasi bahwa proses tuning pada KNN tidak berhasil melampaui performa model Random Forest awal. Dengan demikian, **Random Forest** ditetapkan sebagai model final untuk dievaluasi lebih lanjut.

## Evaluation

Evaluasi final dilakukan pada model dengan performa terbaik, yaitu **model Random Forest dengan parameter default**, untuk mengukur kinerjanya pada data uji.

### Metrik Evaluasi
- **Accuracy**: Rasio prediksi yang benar terhadap total data.
  $$ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} $$
- **Precision**: Dari semua yang diprediksi sakit, berapa persen yang benar-benar sakit.
  $$ \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} $$
- **Recall (Sensitivity)**: Dari semua yang benar-benar sakit, berapa persen yang berhasil terdeteksi.
  $$ \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} $$
- **F1-Score**: Rata-rata harmonik dari Precision dan Recall.
  $$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

### Hasil Evaluasi Model Final
Berikut adalah rekapitulasi hasil dari model **Random Forest** yang terpilih:

| Metric    | Score  |
| :-------- | :----- |
| Accuracy  | 0.8600 |
| Precision | 0.8500 |
| Recall    | 0.9000 |
| F1-Score  | 0.8800 |

### Kesimpulan
Model **Random Forest** menunjukkan performa terbaik dan dipilih sebagai model akhir. Dengan **F1-Score 0.88**, model ini menyajikan keseimbangan yang kuat antara presisi dan recall. Nilai **Recall 90%** sangat krusial, menandakan kemampuan model untuk mengidentifikasi 9 dari 10 pasien yang benar-benar sakit, sehingga meminimalkan risiko kasus positif yang terlewat. Kinerja yang solid ini membuktikan bahwa model dapat menjadi alat bantu yang potensial dan andal untuk skrining awal penyakit jantung.
