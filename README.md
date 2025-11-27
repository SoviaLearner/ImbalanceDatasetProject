# Analisis dan Klasifikasi IPM dengan Machine Learning

Aplikasi untuk menganalisis dan membandingkan model machine learning dalam mengklasifikasikan IPM (Indeks Pembangunan Manusia) pada data yang tidak seimbang (imbalanced), dengan penanganan menggunakan teknik SMOTE.

## 📋 Deskripsi Project

Project ini membahas:
- **Exploratory Data Analysis (EDA)** pada data IPM
- **Penanganan imbalanced data** menggunakan SMOTE
- **Training dan evaluasi** tiga model ML: SVM, KNN, Random Forest
- **Perbandingan performa** model pada data imbalance vs balanced
- **Visualisasi interaktif** menggunakan Streamlit

## 📁 Struktur Folder

\`\`\`
project/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration & constants
├── requirements.txt        # Python dependencies
├── README.md               # Dokumentasi
│
├── data/
│   └── data_loader.py      # Modul untuk loading & preprocessing data
│
├── models/
│   └── model_trainer.py    # Modul untuk training & evaluasi model
│
└── visualization/
    └── plots.py            # Modul untuk visualisasi & plotting
\`\`\`

## 🚀 Cara Menjalankan

### 1. Setup Environment

\`\`\`bash
# Buat virtual environment (opsional tapi recommended)
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Jalankan Aplikasi

\`\`\`bash
streamlit run app.py
\`\`\`

Aplikasi akan membuka di browser pada `http://localhost:8501`

### 3. Upload Dataset

1. Siapkan file Excel dengan sheet bernama "Dataset Imbalance"
2. Kolom yang diperlukan:
   - ID/No (akan dihapus)
   - HLS, PENG, RLS, UHH, IDG, IKK (fitur)
   - IPM (akan dikonversi menjadi Class)

3. Upload file melalui sidebar aplikasi

## 📊 Fitur Aplikasi

### Tab 1: Dataset & Preprocessing
- Tampilkan dataset dan statistik deskriptif
- Penjelasan langkah preprocessing
- Info jumlah training vs test set

### Tab 2: Eksplorasi Data (EDA)
- Distribusi kelas sebelum dan sesudah SMOTE
- Heatmap korelasi antar fitur
- Visualisasi imbalance

### Tab 3: Model Training & Evaluasi
- Training model ML dengan hyperparameter tuning
- Metrik: Akurasi, ROC AUC, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

### Tab 4: Feature Importance
- Analisis kontribusi fitur terhadap prediksi
- Tersedia untuk SVM, KNN, Random Forest
- Visualisasi top N features

### Tab 5: Perbandingan Model
- Perbandingan akurasi ketiga model
- Tabel performa model
- Visualisasi bar chart akurasi

## ⚙️ Konfigurasi

File `config.py` berisi parameter-parameter utama:

- **IPM_BINS & IPM_LABELS**: Klasifikasi IPM ke 4 kategori
- **TEST_SIZE**: Proporsi test set (default 0.2)
- **SMOTE_K_NEIGHBORS**: K neighbors untuk SMOTE (default 5)
- **CV_N_SPLITS, CV_N_REPEATS**: Cross-validation parameters
- **MODEL_PARAMS**: Hyperparameter untuk setiap model

Silakan modifikasi sesuai kebutuhan.

## 🔧 Customization

### Menambah Model Baru

1. Tambahkan di `config.py`:
\`\`\`python
MODEL_PARAMS['NewModel'] = {
    'param1': [values],
    'param2': [values],
}
\`\`\`

2. Tambahkan method di `models/model_trainer.py`:
\`\`\`python
def train_newmodel(self, X_train, y_train):
    # Implementation
    pass
\`\`\`

3. Tambahkan di `app.py`:
\`\`\`python
elif model_option == "NewModel":
    result_train = trainer.train_newmodel(X_train_scaled, y_train)
\`\`\`

### Mengubah Klasifikasi IPM

Edit `config.py`:
\`\`\`python
IPM_BINS = [0, custom_bin1, custom_bin2, float('inf')]
IPM_LABELS = ['Label1', 'Label2', 'Label3', 'Label4']
\`\`\`

## 📚 Teori & Konsep

### Imbalanced Data
Dataset dengan distribusi kelas yang tidak seimbang. Model cenderung bias ke kelas mayoritas.

### SMOTE (Synthetic Minority Oversampling Technique)
Teknik oversampling yang membuat data sintetis pada kelas minoritas menggunakan k-nearest neighbors.

### Model ML
1. **SVM**: Memisahkan kelas dengan hyperplane optimal
2. **KNN**: Klasifikasi berdasarkan jarak ke tetangga terdekat
3. **Random Forest**: Ensemble tree untuk pola kompleks

### Metrik Evaluasi
- **Akurasi**: Proporsi prediksi benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean precision & recall
- **ROC AUC**: Area under curve untuk multi-class

## 📈 Expected Output

Aplikasi akan menampilkan:
- Distribusi kelas imbalance vs balanced
- Metrik evaluasi untuk setiap model
- Confusion matrix
- Feature importance
- Perbandingan performa antar model

## 🛠️ Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **imbalanced-learn**: SMOTE & resampling
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **streamlit**: Web framework

## ⚠️ Troubleshooting

**Error: "No module named 'streamlit'"**
\`\`\`bash
pip install streamlit
\`\`\`

**Error: "Dataset Imbalance sheet not found"**
- Pastikan file Excel memiliki sheet bernama "Dataset Imbalance"
- Format harus .xlsx atau .xls

**Error: "Memory Error" pada SMOTE**
- Dataset terlalu besar atau komputer kekurangan RAM
- Coba reduce dataset atau gunakan sampling yang lebih kecil

## 📝 Notes

- Hyperparameter tuning menggunakan GridSearchCV dengan RepeatedStratifiedKFold
- Semua fitur di-scale menggunakan StandardScaler
- Split stratified untuk mempertahankan distribusi kelas
- Evaluasi menggunakan macro average untuk multi-class metrics

## 📧 Contact & Support

Jika ada pertanyaan atau issue, silakan buat issue di repository.

---

**Last Updated**: November 2025
**Version**: 1.0.0
