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

* Dataset tidak memiliki missing value
* Dataset memiliki 1 duplicate data, yang langsung di drop, sehingga data menjadi 1337.
* Dataset memiliki 9 outlier pada kolom `bmi` tetapi angka `bmi` nya masih masuk akal, dan karena outlier hanya sedikit dibanding jumlah data, maka outlier tidak dibuang.
* Dataset akhir memiliki 1337 data pelanggan.
* Distribusi target (`charges`) menunjukkan skew positif.
* Korelasi terlihat antara `age`, `bmi`, dan `charges`. dengan kolom `age` memiliki korelasi yang lebih tinggi dibanding kolom numerik lain.
* Korelasi juga terlihat pada Fitur `smoker` memiliki dampak besar terhadap `charges`, dengan perokok memiliki biaya lebih tinggi secara signifikan.



## Data Preparation

Tahapan data preparation yang dilakukan:

1. ***Drop Data Duplikat**:

Pada tahap *data preparation*, kami melakukan pemeriksaan terhadap data duplikat menggunakan perintah berikut:

```python
# Mengecek jumlah baris yang terduplikat
insurance_df.duplicated().sum()
```

Hasilnya menunjukkan bahwa terdapat **1 baris duplikat** dalam dataset. Karena keberadaan data duplikat dapat memengaruhi performa model dan menyebabkan bias, maka baris tersebut dihapus menggunakan perintah berikut:

```python
# Menghapus baris duplikat secara permanen
insurance_df.drop_duplicates(inplace=True)
```

> Menghapus data duplikat merupakan langkah penting untuk memastikan bahwa model tidak belajar dari informasi yang berulang, sehingga hasil prediksi menjadi lebih akurat dan tidak bias.


2. **Encoding fitur kategorikal**:
   * Merupakan tahapan untuk mengubah data kategorikal menjadi numerik, karena model hanya menerima input numerik.
   * Menggunakan One-Hot Encoding untuk fitur `region`.
   * Menggunakan Label Encoding untuk fitur `sex`, dan `smoker` karena  binary features.

3. **Split Dataset**:

   * Proses untuk membagi data menjadi train dan test dengan komposisi 80% data train dan 20% data test, ini dilakukan sebelum standarisasi agar tidaka da data leakage ke data test.

4. **Scaling**:

   * Merupakan tahapan yang ditujukan untuk menyeragamkan fitur dengan skala yang sama, sehingga tidak ada fitur yang berat sebelah pada fitur numerik.
   * Scaling dilakukan dengan cara Standarisasi menggunakan Standard Scaler yang akan mengubah data sehingga memiliki mean 0 dan standar deviasi 1.



## Modeling
Menggunakan 3 model, diantaranya :

### Model 1: Linear Regression

#### Cara Kerja

Linear Regression mencari garis lurus terbaik dengan meminimalkan total kuadrat selisih antara prediksi dan nilai aktual.

Persamaan: $y = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$

#### Parameter Default

* `fit_intercept=True`: Menambahkan bias.
* `normalize='deprecated'`: Tidak digunakan lagi di versi baru.
* `copy_X=True`: Menyalin data sebelum fitting.

#### Kelebihan 

* Sederhana dan cepat.
* Mudah diinterpretasi.

#### Kekurangan 

* Hanya menangkap hubungan linear.

### Model 2: Random Forest Regressor

#### Cara Kerja

Membangun banyak decision tree dari subset data yang berbeda, lalu merata-ratakan prediksi dari semua pohon.

#### Parameter Default

* `n_estimators=100`: Jumlah pohon.
* `max_depth=None`: Tidak dibatasi.
* `random_state=42`: Untuk replikasi.
* `criterion='squared_error'`: Fungsi loss.

#### Kelebihan

* Menangani non-linearitas.
* Robust terhadap outlier.

#### Kekurangan 

* Interpretasi kompleks.

### Model 3: XGBoost Regressor

#### Cara Kerja

Model boosting yang memperbaiki kesalahan dari model sebelumnya secara iteratif menggunakan gradient descent.

#### Parameter Default

* `objective='reg:squarederror'`: Fungsi loss.
* `n_estimators=100`: Jumlah boosting rounds.
* `learning_rate=0.3`: Default untuk Langkah pembelajaran.
* `random_state=42`: Konsistensi hasil.

#### Kelebihan 

* Performa tinggi.
* Bisa diatur lewat regularisasi.

#### Kekurangan 

* Waktu training lebih lama.

## Evaluation

Metrik yang digunakan:

* **MAE (Mean Absolute Error)**: Rata-rata selisih absolut.
* **MSE (Mean Squared Error)**: Rata-rata kuadrat selisih.
* **RMSE**: Akar kuadrat MSE.
* **R² Score**: Proporsi variansi yang dijelaskan model.

### Hasil Evaluasi

| Model             | MAE     | MSE         | RMSE    | R² Score |
| ----------------- | ------- | ----------- | ------- | -------- |
| Linear Regression | 4177.05 | 35478020.68 | 5956.34 | 0.8069   |
| Random Forest     | 2676.58 | 22543106.82 | 4747.96 | 0.8773   |
| XGBoost           | 2921.09 | 24782714.67 | 4978.22 | 0.8651   |

### Interpretasi

Model terbaik adalah **Random Forest**, karena memiliki:

* MAE dan RMSE paling rendah
* R² tertinggi (0.8773)

### Feature Importance (Random Forest)

Fitur paling berpengaruh:

1. `smoker_yes`
2. `bmi`
3. `age`

Fitur `smoker_yes` menjadi indikator utama yang memengaruhi prediksi biaya.

---





