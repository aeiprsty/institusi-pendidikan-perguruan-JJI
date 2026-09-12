# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Institusi pendidikan perlu memantau keberhasilan studi mahasiswa karena dropout dapat berdampak pada keberlangsungan pendidikan mahasiswa serta capaian institusi. Mahasiswa yang berhenti dari studi sebelum menyelesaikan pendidikan dapat dipengaruhi oleh berbagai kondisi, seperti performa akademik, kondisi pembayaran biaya pendidikan, karakteristik mahasiswa, maupun kondisi lainnya.

Dataset yang digunakan dalam proyek ini berisi data mahasiswa dengan informasi karakteristik mahasiswa, kondisi akademik, serta status akhir mahasiswa yang terdiri dari `Dropout`, `Enrolled`, dan `Graduate`.

Oleh karena itu, proyek ini dilakukan untuk menganalisis karakteristik mahasiswa berdasarkan status akademiknya, mengidentifikasi faktor yang berkaitan dengan kondisi dropout, menyediakan business dashboard untuk membantu monitoring, serta membangun model machine learning yang dapat digunakan untuk mengidentifikasi mahasiswa yang memiliki risiko dropout berdasarkan data yang tersedia.

### Permasalahan Bisnis

Permasalahan yang perlu diselesaikan dalam proyek ini adalah:

1. Institusi membutuhkan gambaran mengenai distribusi status mahasiswa berdasarkan data yang tersedia.
2. Institusi membutuhkan informasi mengenai karakteristik dan kondisi akademik mahasiswa yang berkaitan dengan status dropout.
3. Institusi membutuhkan media monitoring untuk melihat kondisi mahasiswa berdasarkan faktor akademik, finansial, dan karakteristik lainnya.
4. Institusi membutuhkan sistem yang dapat membantu mengidentifikasi mahasiswa yang memiliki risiko dropout sehingga dapat dilakukan tindak lanjut lebih awal.

### Cakupan Proyek

Proyek ini mencakup:

1. Memahami struktur dan karakteristik dataset mahasiswa.
2. Melakukan data understanding dan data preparation.
3. Melakukan exploratory data analysis untuk menemukan pola yang berkaitan dengan status mahasiswa.
4. Menganalisis faktor akademik, finansial, dan karakteristik mahasiswa.
5. Membuat business dashboard menggunakan Metabase.
6. Membandingkan model Logistic Regression dan Random Forest.
7. Memilih model berdasarkan metrik evaluasi yang relevan dengan kebutuhan identifikasi mahasiswa berisiko dropout.
8. Menyimpan model final untuk digunakan pada sistem prediksi.
9. Membuat prototype prediksi menggunakan Streamlit.
10. Menyusun rekomendasi action items berdasarkan hasil analisis dan pemodelan.

### Persiapan

**Sumber data:** [Students' Performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Dataset terdiri dari **4.424 data mahasiswa dan 37 kolom**, termasuk informasi karakteristik mahasiswa, kondisi akademik, serta status akhir mahasiswa.

Target analisis dibuat dalam bentuk klasifikasi biner dengan ketentuan:

* `Dropout` = 1
* `Enrolled` dan `Graduate` = 0

Target biner digunakan untuk membangun model yang berfokus pada identifikasi mahasiswa yang berstatus `Dropout`.

#### Setup Environment

Proyek ini dikembangkan menggunakan:

* Python 3.11.9
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* SQLite
* Metabase

#### 1. Membuat Virtual Environment

**Windows PowerShell:**

```bash
python -m venv .venv
```

Aktifkan virtual environment:

```bash
.venv\Scripts\Activate.ps1
```

Jika menggunakan Command Prompt:

```bash
.venv\Scripts\activate
```

#### 2. Install Dependency

Setelah virtual environment aktif, jalankan:

```bash
pip install -r requirements.txt
```

#### 3. Menjalankan Notebook

Notebook berisi proses data science mulai dari business understanding, data understanding, data preparation, exploratory data analysis, modeling, evaluation, hingga conclusion.

Jalankan Jupyter Notebook dengan:

```bash
jupyter notebook
```

Kemudian buka file notebook yang terdapat pada folder proyek.

#### 4. Menjalankan Dashboard Metabase

Dashboard dibuat menggunakan Metabase dengan database SQLite.

Pastikan Docker Desktop telah berjalan dan container Metabase telah dibuat. Database SQLite yang digunakan adalah:

```text
student_dropout.db
```

File database ditempatkan pada folder yang di-mount ke container Metabase.

Metabase dapat dijalankan melalui:

```text
http://localhost:3000
```

Setelah Metabase aktif, database SQLite dapat dihubungkan menggunakan path:

```text
/data/student_dropout.db
```

Dashboard yang telah dibuat dapat dibuka melalui koleksi **Personal Collection** pada Metabase.

#### 5. Menjalankan Prototype Streamlit

Model final disimpan dalam file:

```text
dropout_prediction_model.pkl
```

Prototype prediksi terdapat pada:

```text
app.py
```

Jalankan aplikasi dengan:

```bash
streamlit run app.py
```

Setelah aplikasi berjalan, buka alamat yang diberikan oleh Streamlit pada browser.

ATAU dapat mengakses Streamlit Community Cloud berikut:
```
blank
```

## Business Dashboard

Business dashboard dibuat menggunakan **Metabase** dengan judul **Student Dropout Analytics**. Dashboard digunakan untuk memberikan gambaran mengenai distribusi status mahasiswa serta melihat hubungan antara status mahasiswa dengan kondisi akademik, finansial, dan karakteristik mahasiswa.

Dashboard menyediakan beberapa KPI utama:

* Total Students: **4.424**
* Dropout: **1.421**
* Enrolled: **794**
* Graduate: **2.209**

Visualisasi yang digunakan dalam dashboard meliputi:

* Student Status Distribution
* Academic Grade by Student Status
* Tuition Fee Status by Student Status
* Debtor Status by Student Status
* Age at Enrollment by Status
* Approved Units by Status

Dashboard digunakan sebagai media eksplorasi untuk membantu institusi melihat pola yang berkaitan dengan status mahasiswa.

### Insight Utama Dashboard

Berdasarkan hasil analisis, terdapat beberapa temuan utama:

1. Dari 4.424 mahasiswa, sebanyak **2.209 mahasiswa (49,93%)** berstatus Graduate, **1.421 mahasiswa (32,12%)** berstatus Dropout, dan **794 mahasiswa (17,95%)** berstatus Enrolled.
2. Mahasiswa berstatus Dropout memiliki rata-rata jumlah mata kuliah yang disetujui lebih rendah dibandingkan mahasiswa berstatus Enrolled dan Graduate.
3. Rata-rata nilai akademik mahasiswa Dropout juga lebih rendah dibandingkan kelompok Enrolled dan Graduate.
4. Status pembayaran tuition fees menunjukkan perbedaan distribusi antarstatus mahasiswa. Proporsi mahasiswa yang tuition fees-nya belum up to date lebih tinggi pada kelompok Dropout.
5. Status Debtor juga menunjukkan perbedaan distribusi antarstatus mahasiswa.
6. Distribusi usia mahasiswa pada kelompok Dropout cenderung lebih beragam dibandingkan kelompok mahasiswa lainnya.

Temuan tersebut menunjukkan bahwa performa akademik dan beberapa kondisi administratif dapat menjadi indikator yang perlu diperhatikan dalam upaya monitoring risiko dropout.

## Conclusion

Berdasarkan analisis terhadap **4.424 data mahasiswa**, terdapat **1.421 mahasiswa atau 32,12%** yang berstatus Dropout. Sementara itu, sebanyak **794 mahasiswa (17,95%)** berstatus Enrolled dan **2.209 mahasiswa (49,93%)** berstatus Graduate.

Hasil exploratory data analysis menunjukkan adanya perbedaan karakteristik antara mahasiswa Dropout, Enrolled, dan Graduate. Salah satu perbedaan yang paling terlihat terdapat pada performa akademik. Mahasiswa Dropout memiliki rata-rata jumlah mata kuliah yang disetujui serta rata-rata nilai akademik yang lebih rendah dibandingkan kelompok Enrolled dan Graduate.

Pada proses modeling, Logistic Regression dan Random Forest dibandingkan untuk melakukan klasifikasi risiko dropout. Logistic Regression memperoleh accuracy sebesar **88,02%**, precision sebesar **79,67%**, recall sebesar **84,15%**, dan F1-score sebesar **81,85%**. Sementara itu, Random Forest memperoleh accuracy sebesar **87,68%**, precision sebesar **82,05%**, recall sebesar **78,87%**, dan F1-score sebesar **80,43%**.

**Logistic Regression dipilih sebagai model final** karena memiliki recall yang lebih tinggi dibandingkan Random Forest. Recall menjadi metrik yang penting karena model diharapkan dapat mengidentifikasi sebanyak mungkin mahasiswa yang berisiko dropout sehingga dapat menjadi dasar untuk melakukan monitoring dan intervensi lebih awal.

Model yang telah dipilih kemudian disimpan dan digunakan dalam prototype Streamlit untuk melakukan prediksi risiko dropout berdasarkan data mahasiswa yang tersedia.

Hasil prediksi model digunakan sebagai **early warning system**, bukan sebagai dasar pengambilan keputusan secara otomatis. Hasil prediksi tetap perlu dikombinasikan dengan evaluasi dan konteks dari pihak institusi.

### Rekomendasi Action Items

1. **Melakukan monitoring performa akademik secara berkala.** Institusi dapat memberikan perhatian lebih kepada mahasiswa dengan jumlah mata kuliah yang disetujui dan nilai akademik yang rendah.

2. **Menyediakan pendampingan akademik lebih awal.** Mahasiswa yang menunjukkan penurunan performa dapat diarahkan ke program konsultasi akademik, tutoring, atau pendampingan dari dosen pembimbing.

3. **Melakukan monitoring kondisi pembayaran pendidikan.** Mahasiswa dengan kondisi tuition fees yang belum up to date atau memiliki status debtor dapat diberikan informasi dan pendampingan administratif lebih awal.

4. **Menggunakan dashboard sebagai media monitoring.** Dashboard dapat digunakan untuk membantu institusi melihat perubahan distribusi status mahasiswa dan mengidentifikasi kelompok yang membutuhkan perhatian lebih.

5. **Menggunakan model sebagai early warning system.** Model Logistic Regression dapat digunakan untuk membantu mengidentifikasi mahasiswa yang memiliki risiko dropout berdasarkan data yang tersedia. Hasil prediksi sebaiknya digunakan sebagai bahan evaluasi awal dan diikuti dengan tindak lanjut yang sesuai.

6. **Melakukan evaluasi model secara berkala.** Model perlu dievaluasi kembali apabila tersedia data mahasiswa baru agar performanya tetap relevan dengan kondisi institusi.
