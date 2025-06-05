# Laporan Proyek Machine Learning - Sistem Rekomendasi Film [Gibran Malik Naabih Andito]

## Project Overview

Pemilihan film yang sesuai dengan preferensi pribadi pengguna merupakan tantangan tersendiri di tengah banyaknya pilihan film yang tersedia. Sistem rekomendasi menjadi solusi yang dapat membantu pengguna menemukan film yang relevan dan sesuai minat mereka.

Masalah ini penting untuk diselesaikan karena pengalaman pengguna dalam menjelajahi katalog film yang luas akan lebih efisien dan memuaskan dengan adanya rekomendasi yang dipersonalisasi. Beberapa platform besar seperti Netflix dan IMDb telah membuktikan bahwa sistem rekomendasi dapat meningkatkan keterlibatan pengguna.

Referensi:

* Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.

## Business Understanding

### Problem Statements

1. Bagaimana memberikan rekomendasi film kepada pengguna berdasarkan deskripsi dan informasi konten film?
2. Bagaimana memberikan rekomendasi film kepada pengguna berdasarkan pola rating pengguna lain?

### Goals

1. Membangun sistem rekomendasi berbasis konten (Content-Based Filtering) yang dapat memberikan rekomendasi berdasarkan kemiripan konten film.
2. Membangun sistem rekomendasi kolaboratif (Collaborative Filtering) yang dapat memberikan rekomendasi berdasarkan perilaku pengguna lain.

### Solution Approach

Untuk menjawab pernyataan masalah di atas, dua pendekatan yang digunakan:

* **Content-Based Filtering:** Menggunakan TF-IDF dan cosine similarity untuk mencari film yang mirip berdasarkan informasi konten seperti genre, sinopsis, pemeran, sutradara, dan kata kunci.
* **Collaborative Filtering:** Menggunakan pendekatan matrix factorization (SVD) untuk memprediksi rating yang mungkin diberikan pengguna terhadap film yang belum pernah mereka tonton.

## Data Understanding

Dataset yang digunakan adalah "The Movies Dataset" dari TMDB yang tersedia di Kaggle: [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Beberapa file penting dari dataset ini:

* **movies\_metadata.csv** – Informasi utama film (judul, genre, sinopsis, dll)
* **credits.csv** – Informasi pemeran dan kru
* **keywords.csv** – Kata kunci terkait film
* **ratings.csv** – Data rating dari pengguna terhadap film
* **links.csv** – Mapping antar ID di dataset

Beberapa variabel penting:

* `title`: judul film
* `overview`: ringkasan sinopsis film
* `genres`: genre film
* `cast`: pemeran utama
* `crew`: kru (sutradara)
* `keywords`: kata kunci dari film
* `rating`: nilai rating yang diberikan pengguna

Visualisasi distribusi rating dan vote average juga digunakan untuk memahami sebaran data. data awalnya berisi 10000-45000 data

## Data Preparation

Langkah-langkah preprocessing:

* Memfilter dan menggabungkan file movies\_metadata, credits, dan keywords berdasarkan ID
* Mengambil fitur penting: cast, crew (sutradara), genres, dan keywords menggunakan `ast.literal_eval`
* Membuat kolom gabungan (`soup`) untuk digunakan dalam content-based filtering
* Melakukan penghapusan data duplikat pada Movies
* Membersihkan data rating dan memfilter hingga hanya userId ≤ 100 untuk efisiensi
* Membuat pivot table user-item untuk collaborative filtering

Langkah-langkah ini penting untuk menyusun data agar sesuai dengan kebutuhan model rekomendasi.

## Modeling

### Content-Based Filtering

* Menggunakan TF-IDF vectorizer pada kolom `soup`
* Menghitung cosine similarity antar film
* Menghasilkan Top-10 rekomendasi film yang mirip dengan film input pengguna

### Collaborative Filtering

* Menggunakan SVD (Singular Value Decomposition) dari `scipy.sparse.linalg.svds`
* Membentuk matriks rating dan melakukan dekomposisi untuk menghasilkan prediksi rating
* Mengembalikan 10 film dengan prediksi tertinggi untuk masing-masing pengguna

### Kelebihan dan Kekurangan

* **Content-Based**: Tidak bergantung pada user lain, namun bisa membosankan karena terlalu mirip
* **Collaborative**: Lebih personal, tapi tidak bekerja baik untuk pengguna baru (cold start)

## Evaluation

### Metrik Evaluasi: RMSE (Root Mean Squared Error)

Digunakan untuk mengukur perbedaan antara nilai rating yang diprediksi dan rating asli pada data uji.

Formula RMSE:
$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

Hasil evaluasi:

* RMSE pada Collaborative Filtering ≈ *2.1584*

Distribusi rating pengguna juga divisualisasikan untuk mendukung pemahaman terhadap performa model.

## Kesimpulan

* Dua pendekatan sistem rekomendasi berhasil diimplementasikan.
* Content-Based Filtering memberikan hasil yang stabil berdasarkan informasi film.
* Collaborative Filtering memberikan hasil yang lebih adaptif berdasarkan preferensi pengguna.
* Evaluasi menggunakan RMSE menunjukkan bahwa hasil prediksi cukup akurat.
* Sistem dilengkapi dengan fitur interaktif yang dapat menerima input pengguna.

Proyek ini memberikan landasan awal untuk pengembangan sistem rekomendasi yang lebih kompleks dan berskala produksi.
