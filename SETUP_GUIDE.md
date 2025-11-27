# 📖 Panduan Setup dan Instalasi

## Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- File dataset Excel dengan sheet "Dataset Imbalance"

## Langkah-Langkah Setup

### 1. Download/Clone Project

\`\`\`bash
# Jika menggunakan git
git clone <repository-url>
cd project-klasifikasi-ipm

# Atau extract ZIP
unzip project-klasifikasi-ipm.zip
cd project-klasifikasi-ipm
\`\`\`

### 2. Setup Virtual Environment (Recommended)

**Windows:**
\`\`\`bash
python -m venv venv
venv\Scripts\activate
\`\`\`

**macOS/Linux:**
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
\`\`\`

### 3. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Jika ada error, coba install individual:
\`\`\`bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn streamlit openpyxl
\`\`\`

### 4. Verifikasi Instalasi

\`\`\`bash
python -c "import streamlit; print('Streamlit OK')"
python -c "import sklearn; print('Scikit-learn OK')"
python -c "import imblearn; print('Imbalanced-learn OK')"
\`\`\`

### 5. Siapkan Dataset

1. Buat file Excel dengan sheet "Dataset Imbalance"
2. Struktur kolom:
   \`\`\`
   No | HLS | PENG | RLS | UHH | IDG | IKK | IPM
   1  | ... | ...  | ... | ... | ... | ... | ...
   \`\`\`
3. Simpan file (misalnya: `data.xlsx`)

### 6. Jalankan Aplikasi

\`\`\`bash
streamlit run app.py
\`\`\`

Aplikasi akan membuka otomatis di browser pada `http://localhost:8501`

### 7. Upload Dataset

1. Di sidebar, klik "Upload File Excel"
2. Pilih file dataset Anda
3. Tunggu proses loading
4. Eksplorasi tabs untuk analisis

## ✨ Tips

- Gunakan virtual environment untuk menghindari conflict dengan package lain
- Pastikan file Excel sudah di-save dalam format Excel (.xlsx)
- Jika error memory, gunakan dataset yang lebih kecil atau reduce komputation

## 🐛 Common Issues & Solutions

### Issue: "FileNotFoundError: app.py not found"
**Solusi**: Pastikan Anda berada di folder yang benar saat menjalankan command

### Issue: "Sheet 'Dataset Imbalance' not found"
**Solusi**: 
- Buka file Excel
- Pastikan ada sheet dengan nama "Dataset Imbalance"
- Pastikan nama sheet exact match (case-sensitive)

### Issue: ImportError untuk modules
**Solusi**:
\`\`\`bash
# Pastikan path sudah benar
# Atau install ulang dependencies
pip install -r requirements.txt --force-reinstall
\`\`\`

### Issue: Streamlit cache issues
**Solusi**: Clear cache dengan Ctrl+C dan jalankan ulang:
\`\`\`bash
streamlit run app.py --logger.level=debug
\`\`\`

## 📚 Dokumentasi Tambahan

- [Streamlit Docs](https://docs.streamlit.io)
- [Scikit-learn Docs](https://scikit-learn.org)
- [Imbalanced-learn Docs](https://imbalanced-learn.org)

---

Selamat! Setup selesai. Silakan nikmati aplikasi.
\`\`\`

```plaintext file=".gitignore"
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files
*.xlsx
*.xls
*.csv

# Streamlit
.streamlit/
.cache/

# Logs
*.log

# Testing
.pytest_cache/
.coverage

# Distribution
dist/
build/
*.egg-info/
