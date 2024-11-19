# Laporan Proyek Machine Learning 
**Evlin Sitanggang**


## Domain Proyek

Nama *Project* : **Gold Stock Prices Prediction**

**Latar Belakang**:      
Harga emas telah lama menjadi aset investasi andalan karena nilainya yang stabil dan tahan terhadap inflasi. Namun, pergerakan harga emas sangat dipengaruhi oleh berbagai faktor seperti kondisi ekonomi global, tingkat suku bunga, inflasi, dan permintaan pasar. Fluktuasi yang signifikan dalam harga emas sering menjadi tantangan bagi investor yang ingin memprediksi pergerakannya di masa depan.

Dengan kemajuan *Machine Learning (ML)*, analisis data historis kini dapat dilakukan secara lebih mendalam untuk mengidentifikasi pola dan tren dalam pergerakan harga emas. Time series analysis, salah satu metode dalam ML, memungkinkan pemahaman lebih baik terhadap tren jangka panjang, fluktuasi musiman, dan anomali yang ada. Dengan memanfaatkan data historis, pendekatan ini dapat memberikan proyeksi yang lebih akurat tentang harga emas di masa mendatang.

Prediksi berbasis ML ini membantu investor dalam membuat keputusan investasi yang lebih cerdas dengan memanfaatkan wawasan yang dihasilkan dari pola dan tren historis.
Dataset ini menggunakan informasi harga emas dari tahun 2014-2024 dengan rata-rata harga emas per tahun adalah sebagai berikut:

|*Year*| *Gold*|
|------|-------|    
| 2014 | 1240.2|    
| 2015 | 1159.3|    
| 2016 | 1250.9|    
| 2017 | 1260.1|    
| 2018 | 1272.1|    
| 2019 | 1396.4|    
| 2020 | 1779.6|    
| 2021 | 1799.9|    
| 2022 | 1809.1|    
| 2023 | 1953.9|    
| 2024 | 2219.4|    

**Penelitian Terkait**:      
*Windha Mega Pradnya Dhuhita* : 
"Emas adalah logam mulia yang sangat bernilai tinggi, memiliki nilai intrinsik yang signifikan dalam masyarakat modern. Oleh karena itu, semakin banyak orang yang mulai berinvestasi dalam logam mulia yang dikenal sebagai emas. Individu yang ingin berinvestasi dalam emas perlu berhati-hati dalam memantau fluktuasi harga beli dan jual logam mulia ini [[1]](https://www.researchgate.net/publication/376547659_Gold_Price_Prediction_Based_On_Yahoo_Finance_Data_Using_Lstm_Algorithm). Faktor-faktor utama yang memengaruhi harga tersebut meliputi perubahan harga penutupan, harga pembukaan, nilai tertinggi, dan nilai terendah emas. Fenomena yang diamati menunjukkan bahwa harga emas memiliki tingkat volatilitas yang tinggi, terutama akibat fluktuasi yang sering terjadi dan berulang. Teknik *LSTM (Long Short-Term Memory)*, yang dikombinasikan dengan optimasi hiperparameter melalui grid search, memungkinkan prediksi harga emas yang akurat menggunakan data historis harga emas."


## Business Understanding

### Problem Statements

1. Emas adalah komoditas yang sangat bernilai dan serbaguna, digunakan dalam berbagai industri seperti elektronik, dirgantara, kedokteran, dan perhiasan, serta sering dijadikan alat investasi sebagai perlindungan terhadap volatilitas ekonomi. Selain sebagai logam mulia langka, emas juga berfungsi sebagai mata uang dalam transaksi ekonomi global. Harga emas dipengaruhi oleh berbagai faktor, termasuk nilai tukar dolar AS, biaya produksi, inflasi, kebijakan moneter, dan geopolitik, sehingga estimasi harga emas yang akurat sangat penting bagi investor dalam merancang strategi investasi mereka.

2. Sistem prediksi diperlukan untuk memperkirakan pergerakan harga emas berdasarkan data historis, dan metode Long Short-Term Memory (LSTM) cocok digunakan karena kemampuannya memproses informasi jangka panjang dan mengenali pola kompleks serta ketergantungan antar variabel. Dalam prediksi harga emas, LSTM dapat menangani faktor makroekonomi dan tren jangka panjang yang memengaruhi harga. Untuk mengoptimalkan kinerja, penelitian ini akan menggunakan teknik Grid Search untuk mencari kombinasi parameter terbaik, seperti jumlah lapisan, unit per lapisan, dan dropout.

### Goals

1. Meningkatkan kemampuan prediksi harga emas dengan menerapkan algoritma ML yang lebih adaptif, sehingga mampu menangkap dinamika pasar dengan lebih akurat dan efektif dalam berbagai kondisi ekonomi.
    
2. Menyediakan analisis mendalam tentang hubungan antar faktor ekonomi, seperti inflasi dan nilai tukar, yang memengaruhi harga emas, sehingga investor dapat merencanakan strategi investasi dengan lebih baik berdasarkan tren pasar.

### Solution Statements

1. Melatih model LSTM hingga mencapai akurasi terbaik untuk memprediksi nilai harga emas tanpa bergantung pada faktor eksternal yang tidak relevan, sehingga dapat menghasilkan prediksi yang lebih akurat dan handal dalam berbagai kondisi pasar. 

2. Mengujicoba model untuk memprediksi harga emas pada tahun tertentu, dengan tujuan meningkatkan akurasi prediksi dan memberikan estimasi harga emas yang lebih tepat.


## Data Understanding

*Klik link untuk download Dataset*:  
https://www.kaggle.com/datasets/sahilwagh/gold-stock-prices

**Gambaran Umum Dataset**:     
Berkas data ini berformat *Comma Separated Value (CSV)* dengan 2532 baris dan 6 kolom. Berkas ini berisi 5 kolom yang memiliki tipe data numerik, dan 1 kolom dengan tipe data tanggal. Dataset dengan jelas menampilkan nilai variabel-variabel Date, Close, Volume, Open, High, dan Low.

### Variabel-variabel pada dataset adalah sebagai berikut:
-  **Date**    
adalah tanggal yang unik yang menjadi patokan pengukuran harga emas. Tanggal ditulis dalam format *mm/dd/yyyy*.
- **Close/Last**   
adalah harga emas pada akhir perdagangan pada tanggal yang relevan.
- **Volume**   
adalah jumlah volume perdagangan emas pada tanggal yang relevan.
- **Open**    
adalah harga emas pada awal perdagangan pada tanggal yang relevan.
- **High**   
adalah harga tertinggi yang tercatat selama perdagangan emas pada tanggal tertentu.
- **Low**    
adalah harga terendah yang tercatat selama perdagangan emas pada tanggal tertentu.

### Melihat sekilas pada dataset

Sebelum melakukan *modeling* dan *evaluation* dengan algoritma *Machine Learning* pada Notebook "Google Colaboratory", mari melihat sekilas pada *dataset* yang digunakan. Di bawah ini akan ditampilkan cuplikan dari dataset, deskripsi statistik dan korelasi antar variabel beserta penjelasannya.


Tabel 1 : Cuplikan Dataset  
 
|Date        | Close/Last | Volume   | Open    | High    | Low   |
|------------|------------|----------|---------|---------|-----	 |
|2024-07-03  | 2369.4     | 185930.0 | 2338.6  | 2374.5  | 2335.7|
|2024-07-02  | 2333.4     | 146568.0 | 2341.6  | 2346.1  | 2327.4|
|2024-07-01  | 2338.9     | 136861.0 | 2336.2  | 2348.8  | 2328.2|
|2024-06-28  | 2339.6     | 131191.0 | 2338.6  | 2350.6  | 2329.7|
|2024-06-27  | 2336.6     | 135784.0 | 2309.4  | 2342.0  | 2306.8|
|...         | ...        | ...      | ...     | ...     | ...	  |
|2014-07-11  | 1337.4     | 88470.0  | 1336.5  | 1340.4  | 1334.6|
|2014-07-10  | 1339.2     | 167391.0 | 1325.0  | 1346.8  | 1325.0|
|2014-07-09  | 1324.3     | 155101.0 | 1320.4  | 1333.4  | 1318.7|
|2014-07-08  | 1316.5     | 126706.0 | 1320.8  | 1325.7  | 1314.3|
|2014-07-07  | 1317.0     | 79110.0  | 1321.4  | 1321.7  | 1312.1|

Tabel 1 menyajikan 5 baris pertama dan 5 baris terakhir pada dataset yang belum dilakukan *preprocessing*.


Tabel 2 : Deskripsi Informasi Dataset      
Data columns (total 6 columns):
| #  |Column     |Non-Null Count  |Dtype  |
|--- |------     |--------------  |-----  |
|0   |Unnamed: 0 |2511 non-null   |int64  |
|1   |Close/Last |2511 non-null   |float64|
|2   |Volume     |2511 non-null   |float64|
|3   |Open       |2511 non-null   |float64|
|4   |High       |2511 non-null   |float64|
|5   |Low        |2511 non-null   |float64|

dtypes: float64(5), int64(1)
memory usage: 137.3 KB

Tabel 2 memberikan informasi mengenai statistik *dataset* yaitu Data ini terdiri dari 2511 entri dengan 6 kolom yang mencakup informasi harga saham dan volume perdagangan, yaitu Close/Last, Volume, Open, High, dan Low. Semua kolom berisi data non-null dengan tipe data numerik, yakni float64 untuk harga dan volume, serta integer untuk kolom pertama yang kemungkinan merupakan indeks. Data ini tidak memiliki nilai yang hilang dan menggunakan memori sebesar 137.3 KB.


Tabel 3 : Korelasi Antar Variabel    
   
| Unnamed: 0    | Close/Last  | Volume      | Open        | High        | Low         |
|---------------|-------------|-------------|-------------|-------------|-------------|
| Unnamed: 0    | 1.000000    | -0.914928   | -0.056636   | -0.914731   | -0.913637   |
| Close/Last    | -0.914928   | 1.000000    | 0.019551    | 0.999135    | 0.999589    |
| Volume        | -0.056636   | 0.019551    | 1.000000    | 0.023158    | 0.027213    |
| Open          | -0.914731   | 0.999135    | 1.000000    | 0.999544    | 0.999473    |
| High          | -0.913637   | 0.999589    | 0.027213    | 1.000000    | 0.999369    |
| Low           | -0.916239   | 0.999644    | 0.015087    | 0.999473    | 1.000000    |

Berikut adalah analisis korelasi antar variabel dalam dataset seperti yang ditampilkan tabel 3:
1. Korelasi antara kolom *"Close/Last"* dan *"Open"* sangat kuat, dengan nilai korelasi mencapai 0.999135. Hal ini menunjukkan bahwa harga penutupan (Close/Last) memiliki hubungan yang hampir identik dengan harga pembukaan (Open), yang biasa terjadi di pasar saham di mana harga penutupan sebelumnya seringkali mempengaruhi harga pembukaan pada hari berikutnya.

2. Korelasi antara *"Close/Last"* dan *"High"* serta *"Close/Last"* dan *"Low"* juga sangat tinggi, dengan nilai korelasi 0.999589 dan 0.999644, masing-masing. Ini mengindikasikan bahwa harga tertinggi (High) dan harga terendah (Low) dalam satu hari cenderung sejalan dengan harga penutupan. Hubungan ini menunjukkan bahwa pergerakan harga dalam satu periode (misalnya sehari) cukup konsisten, meskipun harga pembukaan dan penutupan bisa berbeda.

3. Volume memiliki korelasi yang rendah dengan semua kolom harga (hanya 0.019551 dengan *"Close/Last"* dan 0.023158 dengan *"Open"*). Ini menunjukkan bahwa volume perdagangan tidak banyak memengaruhi harga saham secara langsung dalam dataset ini. Pergerakan harga lebih dipengaruhi oleh faktor lain selain volume perdagangan.

4. Korelasi antara *"Open"*, *"High"*, dan *"Low"* menunjukkan hubungan yang sangat kuat, di atas 0.999, mengindikasikan bahwa harga pembukaan cenderung dekat dengan harga tertinggi dan terendah pada hari tersebut, yang menunjukkan volatilitas harga yang relatif kecil dalam satu periode waktu.

Secara keseluruhan, data menunjukkan bahwa harga penutupan memiliki korelasi yang sangat kuat dengan harga pembukaan, tertinggi, dan terendah, sementara volume perdagangan memiliki pengaruh yang sangat terbatas terhadap pergerakan harga.
  

## Data Preparation

**Tahapan dalam Persiapan Data**
1. Memilih Kolom untuk Prediksi
Pada tahap pertama, kolom `Close/Last` dipilih dari DataFrame (`df`) sebagai data yang akan diprediksi.
- `df['Close/Last']` mengambil kolom "Close/Last" yang berisi harga penutupan.
- Data tersebut kemudian diubah menjadi DataFrame (`pd.DataFrame(dataset)`).
- `data = dataset.values` mengonversi DataFrame menjadi array numpy agar bisa digunakan dalam pemrosesan lebih lanjut.

2. Normalisasi Data
Pada tahap ini, data harga penutupan dinormalisasi ke rentang [0, 1] menggunakan `MinMaxScaler` dari `sklearn`, yang penting untuk memastikan skala data seragam, terutama pada model berbasis jaringan saraf seperti LSTM.
- `MinMaxScaler(feature_range=(0, 1))` menginisialisasi scaler untuk merubah data menjadi rentang antara 0 dan 1.
- `np.array(data).reshape(-1, 1)` mengubah data menjadi array satu dimensi agar dapat dinormalisasi.
- `scaler.fit_transform()` digunakan untuk menormalkan data dengan skala yang telah ditentukan.

3. Pembagian Data Menjadi Data Latih dan Data Uji
Pada tahap ini, data dibagi menjadi dua bagian: data latih (80%) dan data uji (20%). Pembagian ini digunakan untuk melatih dan menguji model.
- `train_size` dihitung sebagai 80% dari total data, sedangkan `test_size` adalah 20%.
- `train_data` berisi data latih yang diambil dari `scaled_data`, sementara `test_data` adalah data uji yang mengambil 60 data terakhir dari data latih.

4. Membuat Set Pelatihan
Di tahap ini, data pelatihan dipersiapkan untuk LSTM dengan struktur time-series, di mana setiap input berisi 60 waktu sebelumnya untuk memprediksi harga selanjutnya.
- Loop dimulai dari indeks ke-60 untuk mengambil 60 waktu sebelumnya sebagai input.
- `x_train` berisi data input yang terdiri dari 60 waktu sebelumnya, sementara `y_train` berisi nilai target (harga setelahnya).
- Setelah loop, `x_train` dan `y_train` diubah menjadi array numpy.

5. Menyesuaikan Bentuk Data untuk LSTM
LSTM memerlukan input dalam bentuk tiga dimensi, yaitu `(samples, time_steps, features)`. Oleh karena itu, data latih diubah bentuknya agar sesuai dengan format tersebut.
- `x_train.shape[0]` adalah jumlah sampel, `x_train.shape[1]` adalah jumlah waktu (60), dan `1` adalah jumlah fitur (karena hanya menggunakan satu fitur, yaitu harga penutupan).
- `np.reshape` digunakan untuk mengubah bentuk `x_train` menjadi tiga dimensi yang sesuai dengan input LSTM.


## Modeling

### 1. LSTM
LSTM adalah jenis jaringan syaraf tiruan (neural network) yang dirancang untuk menangani data urutan (sequential data), seperti data waktu (time-series).

**Cara Kerja**: 
1. LSTM Cell Structure: LSTM terdiri dari beberapa komponen penting yang disebut gates:
2. Forget Gate: Memutuskan informasi mana yang akan dilupakan dari memori sebelumnya.
3. Input Gate: Memutuskan informasi baru mana yang akan ditambahkan ke dalam memori.
4. Cell State: Menyimpan informasi jangka panjang yang relevan.
5. Output Gate: Memutuskan apa yang akan dikeluarkan dari memori LSTM pada waktu tertentu.
6. Pengolahan Urutan Data: LSTM memproses data sekuensial satu langkah pada suatu waktu, meng-update memori dan menyesuaikan output berdasarkan informasi yang diterima sebelumnya.

**Kelebihan LSTM**
1. Mengatasi Vanishing Gradient: LSTM mampu mengatasi masalah vanishing gradient yang terjadi pada RNN tradisional dengan menyimpan informasi dalam jangka panjang.
2. Memori Jangka Panjang: Dapat mengingat informasi penting dalam jangka waktu yang lebih lama, yang sangat berguna untuk data urutan yang bergantung pada konteks jauh (seperti prediksi saham).
3. Fleksibel: Bisa digunakan untuk berbagai masalah prediksi data urutan, baik yang memiliki pola musiman atau tren yang panjang.
4. Pemrosesan Paralel: LSTM memungkinkan pemrosesan paralel meskipun data diberikan dalam urutan waktu.

**Kekurangan LSTM**
1. Kompleksitas dan Waktu Pelatihan: LSTM cenderung lebih lambat untuk dilatih dibandingkan dengan model lain seperti jaringan feed-forward karena kompleksitasnya yang lebih tinggi.
2. Perlu Data Dalam Jumlah Besar: LSTM membutuhkan data yang besar untuk mencapai performa terbaiknya.
3. Overfitting: LSTM, jika tidak ditangani dengan benar, bisa mengalami overfitting jika model terlalu kompleks untuk dataset kecil.
4. Kesulitan dalam Menangani Data yang Sangat Panjang: Meskipun lebih baik daripada RNN tradisional, LSTM masih dapat kesulitan mengelola data dengan urutan yang sangat panjang. 

**Parameter yang Digunakan pada LSTM**
1. *Jumlah Neuron pada Layer LSTM*: Pada model di atas, terdapat dua layer LSTM, masing-masing dengan 50 dan 64 unit. Parameter ini menentukan banyaknya "unit memori" dalam LSTM yang mengontrol kemampuan model untuk memproses data.

2. *Return Sequences*: Pada LSTM pertama (LSTM(50, return_sequences=True)), return_sequences=True berarti output dari layer LSTM pertama adalah urutan (sequential output), yang kemudian diteruskan ke LSTM kedua. Ini diperlukan jika kita ingin menggunakan lebih dari satu layer LSTM berurutan.

3. *Epochs*: Pada model ini, training dilakukan selama 100 epoch, yang berarti model akan melakukan iterasi melalui seluruh data 100 kali untuk memperbarui bobot dan biasnya.

4. *Batch Size*: Batch size adalah ukuran subset data yang digunakan untuk melatih model pada satu waktu. Model ini menggunakan batch size sebesar 128, yang mempengaruhi kecepatan pelatihan dan penggunaan memori.

5. *EarlyStopping*: Menghentikan pelatihan lebih awal jika model tidak menunjukkan peningkatan pada metrik yang dipilih, dalam hal ini, loss. Hal ini mencegah model mengalami overfitting dan mengurangi waktu pelatihan yang tidak diperlukan.

6. *Dropout*: Dropout digunakan untuk mencegah overfitting dengan cara secara acak menonaktifkan beberapa neuron selama pelatihan.

**Hasil Training**
Selama proses pelatihan, model menunjukkan penurunan loss yang konsisten, dengan nilai loss terkecil mencapai *2.1884e-04* pada epoch ke-99.
Nilai *mean absolute error (MAE)* juga menurun, dengan nilai MAE terakhir pada epoch ke-100 adalah 0.0109, yang menunjukkan bahwa kesalahan rata-rata prediksi model pada data pelatihan sangat kecil.

## Evaluation
 
### Metrik Evaluasi dan penjelasannya
 Untuk mengevaluasi akurasi prediksi harga emas, sejumlah metrik evaluasi yang dapat digunakan adalah:
1. **Mean Absolute Error (MAE)**:    
    * *Formula*: MAE = (1/n) * Σ |*actual - predicted*|
    * MAE mengukur rata-rata dari selisih absolut antara nilai aktual dan nilai prediksi oleh model.
    * Semakin rendah nilai MAE, semakin baik kinerja model karena prediksi lebih mendekati nilai aktual.  
    * MAE berguna ketika outlier (nilai yang jauh dari rata-rata) dalam data tidak memiliki dampak yang besar pada evaluasi model.

### Memilih model yang terbaik berdasarkan metrik evaluasi

16/16 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - *loss*: 1.8702e-04 - *mean_absolute_error*: 0.0108
16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Pada data uji, model menunjukkan performa yang sangat baik dengan menghasilkan loss yang sangat rendah, yaitu *1.8702e-04*, yang mengindikasikan bahwa model memiliki kemampuan untuk meminimalkan kesalahan prediksi secara efektif. Selain itu, mean absolute error (MAE) sebesar 0.0108 menunjukkan bahwa rata-rata selisih antara harga emas yang diprediksi dengan harga emas yang sebenarnya hanya sekitar *1.08%*, yang tergolong sangat akurat. Hasil ini mengindikasikan bahwa model *LSTM* yang dilatih mampu memberikan prediksi yang sangat dekat dengan nilai yang sesungguhnya, menandakan kemampuannya dalam menangkap pola dan tren dalam data harga emas secara efektif.


**Kesimpulan**:
**MAE (Mean Absolute Error)**:    
  MAE yang *lebih rendah* adalah yang lebih baik. Semakin rendah nilai MAE, semakin baik model dalam melakukan prediksi.

Berdasarkan hasil permodelan dan evaluasi, model LSTM disimpulkan cukup baik untuk diterapkan pada dataset prediksi emas.

---

**Referensi**:

* [[1]](https://www.researchgate.net/publication/376547659_Gold_Price_Prediction_Based_On_Yahoo_Finance_Data_Using_Lstm_Algorithm) Dhuhita, Windha & Farid, Muhammad & Yaqin, Ainul & Haryoko, Haryoko & Huda, Arif. (2023). Gold Price Prediction Based On Yahoo Finance Data Using Lstm Algorithm. 420-425. 10.1109/ICIMCIS60089.2023.10349035.
 


**---Ini adalah bagian akhir laporan---**



