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

Jumlah Kolom dan baris

* **movies\_metadata.csv** – 10000 baris dan 24 kolom
* **credits.csv** – 45476 baris dan 3 kolom
* **keywords.csv** – 46419 baris dan 2 kolom
* **ratings.csv** – 98929 baris dan 4 kolom
* **links.csv** – 45843 baris dan 3 kolom

terdapat nilai nilai yang hilang, seperti :

- belongs_to_collection	: 8579
- homepage	: 9338
- imdb_id	: 1
- overview	: 29
- poster_path	: 31
- release_date	: 5
- runtime	: 6
- status	: 8
- tagline	: 3108
- tmdbId	: 219

Beberapa variabel penting:

* `title`: judul film
* `overview`: ringkasan sinopsis film
* `genres`: genre film
* `cast`: pemeran utama
* `crew`: kru (sutradara)
* `keywords`: kata kunci dari film
* `rating`: nilai rating yang diberikan pengguna

berikut isi dari dataset utamanya :

- adult: Menunjukkan apakah film ditujukan untuk penonton dewasa (konten eksplisit). Bernilai True atau False.

- belongs_to_collection: Informasi apakah film merupakan bagian dari suatu koleksi film/franchise. Jika ada, biasanya berbentuk string JSON (berisi nama koleksi, ID, dll).

- budget: Anggaran produksi film dalam satuan USD. Tipe data numerik (int64).

- genres: Genre film dalam bentuk daftar objek JSON. Contoh: [{"id": 28, "name": "Action"}].

- homepage: URL resmi dari film tersebut, jika tersedia.

- id: ID unik film dalam database (biasanya dari TMDb). Tipe numerik (int64).

- imdb_id: ID film pada situs IMDb (Internet Movie Database), digunakan untuk pencocokan lintas platform.

- original_language: Kode bahasa asli film (contoh: 'en' untuk Inggris, 'fr' untuk Prancis).

- original_title: Judul asli film, sesuai bahasa aslinya.

- overview: Ringkasan atau sinopsis singkat dari film.

- popularity: Skor popularitas berdasarkan metrik TMDb yang mencakup view, like, dan interaksi lainnya.

- poster_path: Path (alamat relatif) ke poster film, biasanya dapat digunakan bersama dengan domain TMDb untuk menampilkan gambar.

- production_companies: Daftar rumah produksi yang membuat film. Bentuknya biasanya string JSON.

- production_countries: Negara tempat produksi film dilakukan. Bentuknya juga string JSON.

- release_date: Tanggal rilis resmi film (format: YYYY-MM-DD).

- revenue: Pendapatan kotor yang dihasilkan film di seluruh dunia (dalam USD).

- runtime: Durasi film dalam satuan menit.

- spoken_languages: Bahasa yang digunakan dalam film. Format: daftar objek JSON dengan kode dan nama bahasa.

- status: Status rilis film. Contoh: 'Released', 'Post Production', 'Rumored'.

- tagline: Slogan atau kalimat promosi pendek dari film (sering muncul di poster).

- title: Judul film (bisa berbeda dari original_title jika diterjemahkan untuk distribusi internasional).

- video: Bernilai True jika entri mengacu pada konten video (biasanya selalu False untuk film reguler).

- vote_average: Rata-rata skor rating dari pengguna TMDb terhadap film tersebut.

- vote_count: Jumlah total pengguna yang memberikan rating terhadap film.

Visualisasi distribusi rating dan vote average juga digunakan untuk memahami sebaran data.

## Data Preparation

Langkah-langkah preprocessing:

* Membersihkan data rating dan memfilter hingga hanya userId ≤ 1000 untuk efisiensi
* Memfilter dan menggabungkan file movies\_metadata, credits, dan keywords berdasarkan ID, serta menghapus overview yang kosong.
* mengubah id menjadi string sebelum di jadikan 1
* Mengambil fitur penting: cast, crew (sutradara), genres, dan keywords menggunakan `ast.literal_eval`
* Membuat kolom gabungan (`soup`) untuk digunakan dalam content-based filtering
* Melakukan konversi kepada beberapa list (cast, genres, keywords) menjadi tuple movies
* Melakukan penghapusan data duplikat pada Movies
* penghapusan nilai kosong pada tmdbId, konversi tipe data tmdbId, dan proses grouping (groupby) untuk mendapatkan rata-rata rating per user-item.
* Membuat pivot table user-item untuk collaborative filtering
* Menggunakan TF-IDF vectorizer pada kolom `soup`

Langkah-langkah ini penting untuk menyusun data agar sesuai dengan kebutuhan model rekomendasi.

## Modeling

### Content-Based Filtering

* Menghitung cosine similarity antar film
* Menghasilkan Top-10 rekomendasi film yang mirip dengan film input pengguna

### Collaborative Filtering

* Menggunakan SVD (Singular Value Decomposition) dari `scipy.sparse.linalg.svds`
* Membentuk matriks rating dan melakukan dekomposisi untuk menghasilkan prediksi rating
* Mengembalikan 10 film dengan prediksi tertinggi untuk masing-masing pengguna

### Kelebihan dan Kekurangan

* **Content-Based**: Tidak bergantung pada user lain, namun bisa membosankan karena terlalu mirip

contoh rekomendasi 'Toy story' :

'Toy Story 2', 'Small Soldiers', 'The Champ', 'Toys', 'Everything You Always Wanted to Know About Sex *But Were Afraid to Ask', 'Dolls', 'Take the Money and Run', 'The Transformers: The Movie', 'Stardust Memories', "Child's Play 3"

* **Collaborative**: Lebih personal, tapi tidak bekerja baik untuk pengguna baru (cold start)

contoh rekomendasi untuk pengguna 10 :

'The Shawshank Redemption', 'Forrest Gump', 'Jurassic Park', 'Terminator 2: Judgment Day', 'Outbreak', 'GoldenEye', 'The Lion King', 'Braveheart', 'Crimson Tide', 'Speed'

## Evaluation

### Metrik Evaluasi Content-Based : Precision@K, dan Recall@K

setelah dihitung menghasilkan nilai rekomendasi content based untuk 'Toy Story':

Precision : 0,8

Recall : 0,0021

F1-Score : 0,0041

### Metrik Evaluasi Collaborative: RMSE (Root Mean Squared Error)

Digunakan untuk mengukur perbedaan antara nilai rating yang diprediksi dan rating asli pada data uji.

Formula RMSE:
$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

Hasil evaluasi:

* RMSE pada Collaborative Filtering ≈ *2.1458*

Distribusi rating pengguna juga divisualisasikan untuk mendukung pemahaman terhadap performa model.

## Kesimpulan

* Dua pendekatan sistem rekomendasi berhasil diimplementasikan.
* Content-Based Filtering memberikan hasil yang stabil berdasarkan informasi film.
* Collaborative Filtering memberikan hasil yang lebih adaptif berdasarkan preferensi pengguna.
* Evaluasi menggunakan RMSE menunjukkan bahwa hasil prediksi cukup akurat.
* Sistem dilengkapi dengan fitur interaktif yang dapat menerima input pengguna.

Proyek ini memberikan landasan awal untuk pengembangan sistem rekomendasi yang lebih kompleks dan berskala produksi.
