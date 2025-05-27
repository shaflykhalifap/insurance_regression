# Laporan Proyek Machine Learning - Shafly Khalifa Pamungkas

## Domain Proyek

Industri asuransi kesehatan telah mengalami pertumbuhan signifikan dalam beberapa tahun terakhir, terutama karena meningkatnya kesadaran akan pentingnya proteksi terhadap risiko kesehatan. Dalam industri ini, penentuan premi atau biaya asuransi kesehatan menjadi salah satu elemen penting yang memengaruhi keputusan pelanggan dan keuntungan perusahaan.

Penetapan biaya premi biasanya mempertimbangkan berbagai faktor seperti usia, jenis kelamin, status merokok, indeks massa tubuh (BMI), jumlah tanggungan anak, dan wilayah tempat tinggal. Namun, proses ini seringkali kompleks dan dapat menimbulkan bias atau ketidakakuratan apabila hanya dilakukan secara manual.

Dengan demikian, penggunaan teknologi machine learning dapat menjadi solusi yang efektif untuk memprediksi biaya asuransi secara lebih akurat dan adil. Pendekatan ini memungkinkan perusahaan asuransi untuk mengotomatiskan proses penentuan premi berdasarkan data historis pelanggan.

Masalah ini menjadi krusial, karena apabila perusahaan asuransi menetapkan harga premi yang salah, maka akan mengakibatkan kerugian perusahaan tersebut sendiri, karena yang ditanggung lebih dari apa yang seharusnya.

Maka dari itu solusi ini hadir untuk menentukan harga yang tepat sesuai kategori kategori yang ada.

**Referensi:**

* Cramer, J. S., "The Origins of Logistic Regression," *Tinbergen Institute Discussion Paper*, 2002.
* American Academy of Actuaries. "Risk Classification Statement of Principles," 2011.
* Datasets: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)

## Business Understanding

### Problem Statements

* Bagaimana memprediksi biaya asuransi kesehatan berdasarkan data pelanggan?
* Fitur mana yang paling mempengaruhi penentuan biaya asuransi?

### Goals

* Mengembangkan model prediktif untuk estimasi biaya asuransi berdasarkan fitur pelanggan.
* Mengidentifikasi fitur yang paling signifikan dalam mempengaruhi biaya asuransi.

### Solution Statements

* Membangun beberapa model regresi: Linear Regression, Random Forest Regressor, dan XGBoost Regressor.
* Menggunakan metrik evaluasi seperti MAE, MSE, RMSE, dan R² Score untuk membandingkan performa model.
* Menentukan model terbaik berdasarkan nilai evaluasi dan interpretasi fitur importance dari model pohon.

## Data Understanding

Dataset yang digunakan adalah *Medical Cost Personal Dataset* dari Kaggle.

**Link Dataset**: [https://www.kaggle.com/mirichoi0218/insurance](https://www.kaggle.com/mirichoi0218/insurance)

Dataset ini terdiri dari 1338 data pelanggan dengan fitur-fitur:

* `age`: usia pelanggan (numerik)
* `sex`: jenis kelamin (kategori: male/female)
* `bmi`: indeks massa tubuh (numerik)
* `children`: jumlah anak tanggungan (numerik)
* `smoker`: status merokok (kategori: yes/no)
* `region`: wilayah tempat tinggal (kategori: northeast, northwest, southeast, southwest)
* `charges`: biaya asuransi (target variabel)

### Exploratory Data Analysis (EDA)

* Distribusi target (`charges`) menunjukkan skew positif.
* Fitur `smoker` memiliki dampak besar terhadap `charges`, dengan perokok memiliki biaya lebih tinggi secara signifikan.
* Korelasi juga terlihat antara `age`, `bmi`, dan `charges`.

## Data Preparation

Tahapan data preparation yang dilakukan:

1. **Handling categorical data**:
   * Menggunakan Label Encoding untuk fitur `sex`, dan `smoker`
   * Menggunakan One-Hot Encoding untuk fitur `region`.

2. **Feature scaling**:

   * Menggunakan `StandardScaler` pada fitur numerik: `age`, `bmi`, `children`.

3. **Train-Test Split**:

   * Membagi data menjadi 80% data latih dan 20% data uji menggunakan `train_test_split()`.

Alasan dilakukan scaling adalah untuk memastikan fitur numerik berada dalam skala yang sama, agar model seperti Linear Regression dapat bekerja optimal.

## Modeling

Tiga model regresi digunakan:

### 1. Linear Regression

* Kelebihan: Mudah diinterpretasikan, cepat.
* Kekurangan: Tidak menangkap non-linearitas.

### 2. Random Forest Regressor

* Kelebihan: Robust terhadap outlier, mampu menangkap interaksi non-linear.
* Kekurangan: Interpretasi kompleks, bisa overfitting.

### 3. XGBoost Regressor

* Kelebihan: Performa tinggi, penanganan regularisasi lebih baik.
* Kekurangan: Butuh tuning parameter agar optimal.

Model terbaik dipilih berdasarkan metrik evaluasi.

## Evaluation

Metrik evaluasi yang digunakan:

* **MAE (Mean Absolute Error)**: Rata-rata kesalahan absolut.
* **MSE (Mean Squared Error)**: Rata-rata kesalahan kuadrat.
* **RMSE**: Akar dari MSE, dalam satuan yang sama dengan target.
* **R² Score**: Koefisien determinasi, semakin mendekati 1 berarti semakin baik.

### Hasil Evaluasi

| Model             | MAE     | MSE         | RMSE    | R² Score |
| ----------------- | ------- | ----------- | ------- | -------- |
| Linear Regression | 4177.05 | 35478020.68 | 5956.34 | 0.8069   |
| Random Forest     | 2676.58 | 22543106.82 | 4747.96 | 0.8773   |
| XGBoost           | 2921.09 | 24782714.67 | 4978.22 | 0.8651   |

### Interpretasi

Random Forest menghasilkan performa terbaik dengan MAE dan RMSE paling rendah serta R² tertinggi.

### Feature Importance (Random Forest)

Fitur-fitur paling berpengaruh:

* `smoker_yes`
* `bmi`
* `age`

Fitur `smoker_yes` memberikan kontribusi paling besar terhadap prediksi biaya asuransi, disusul oleh `bmi` dan `age`.

---

*Proyek ini menunjukkan bahwa model Random Forest paling efektif dalam memprediksi biaya asuransi berdasarkan data historis pelanggan. Insight yang diperoleh dari model juga dapat digunakan untuk memahami faktor risiko terbesar dalam penetapan biaya.*
