"""
Configuration file untuk project klasifikasi IPM.
Menyimpan konstanta, parameter model, dan warna SDGs yang digunakan di seluruh project.
"""

# ==================== Konstanta Klasifikasi IPM ====================
IPM_BINS = [0, 60, 69.9, 79.9, float('inf')]
IPM_LABELS = ['Rendah', 'Menengah', 'Tinggi', 'Sangat Tinggi']

# ==================== Parameter Train-Test Split ====================
TEST_SIZE = 0.2
RANDOM_STATE = 123

# ==================== Parameter SMOTE ====================
SMOTE_SAMPLING_STRATEGY = 'not majority'
SMOTE_K_NEIGHBORS = 5

# ==================== Parameter Model ====================
MODEL_PARAMS = {
    'SVM': {
        'kernel': 'rbf',
        'C_values': [0.1, 1, 10, 100],
        'probability': True,
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
    },
    'Random Forest': {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20, None],
    }
}

# ==================== Cross-Validation Parameters ====================
CV_N_SPLITS = 5
CV_N_REPEATS = 3

# ==================== File Paths ====================
DATA_SHEET_NAME = 'Dataset Imbalance'

# ==================== Warna SDGs untuk Visualisasi ====================
SDGS_COLORS = {
    "No Poverty": "#E5243B",
    "Good Health & Well-being": "#4C9F38",
    "Quality Education": "#C5192D",
    "Gender Equality": "#FF3A21",
    "Clean Water & Sanitation": "#26BDE2",
    "Affordable & Clean Energy": "#FCC30B",
    "Decent Work & Economic Growth": "#A21942",
    "Industry, Innovation & Infrastructure": "#FD6925",
    "Reduced Inequalities": "#DD1367",
    "Sustainable Cities & Communities": "#FD9D24",
    "Responsible Consumption & Production": "#BF8B2E",
    "Climate Action": "#3F7E44",
    "Life Below Water": "#0A97D9",
    "Life on Land": "#56C02B",
    "Peace, Justice & Strong Institutions": "#00689D",
    "Partnerships for the Goals": "#19486A"
}

SDGS_CLASS_COLORS = SDGS_COLORS  # untuk label/class
SDGS_PLOT_COLORS = list(SDGS_COLORS.values())  # untuk plotting
