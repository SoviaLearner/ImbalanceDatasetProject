"""
Module untuk memuat dan preprocessing data IPM.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import (
    IPM_BINS, IPM_LABELS, TEST_SIZE, RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY, SMOTE_K_NEIGHBORS, DATA_SHEET_NAME
)


class DataLoader:
    """Class untuk loading dan preprocessing data IPM."""
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data_original = None
        self.data_processed = None
        
    def load_data(self, file_path):
        """
        Memuat data dari file Excel.
        
        Parameters:
        -----------
        file_path : str
            Path ke file Excel.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame yang sudah diload.
        """
        self.data_original = pd.read_excel(file_path, sheet_name=DATA_SHEET_NAME)
        return self.data_original.copy()
    
    def create_class_column(self, data, ipm_column='IPM'):
        """
        Membuat kolom Class berdasarkan nilai IPM.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame input.
        ipm_column : str
            Nama kolom IPM.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan kolom Class ditambahkan.
        """
        data_copy = data.copy()
        data_copy['Class'] = pd.cut(
            data_copy[ipm_column],
            bins=IPM_BINS,
            labels=IPM_LABELS,
            right=False
        )
        return data_copy
    
    def remove_irrelevant_columns(self, data):
        """
        Menghapus kolom ID dan IPM dari data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame input.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame dengan kolom irrelevant dihapus.
        """
        # Hapus kolom pertama (ID/No) dan kolom 'IPM'
        columns_to_drop = [data.columns[0], 'IPM']
        return data.drop(columns=columns_to_drop)
    
    def apply_smote(self, X, y):
        """
        Menerapkan SMOTE untuk balancing data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
            
        Returns:
        --------
        tuple
            (X_balanced, y_balanced)
        """
        smote = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=self.random_state
        )
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
    
    def split_data(self, X, y, use_smote=True):
        """
        Melakukan splitting dan scaling data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        use_smote : bool
            Apakah menggunakan SMOTE atau tidak.
            
        Returns:
        --------
        dict
            Dictionary berisi X_train_scaled, X_test_scaled, y_train, y_test.
        """
        # Apply SMOTE jika diperlukan
        if use_smote:
            X_balanced, y_balanced = self.apply_smote(X, y)
            stratify_y = y_balanced
        else:
            X_balanced = X.copy()
            y_balanced = y.copy()
            stratify_y = y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced,
            test_size=TEST_SIZE,
            stratify=stratify_y,
            random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        return {
            'X_train': X_train_scaled_df,
            'X_test': X_test_scaled_df,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler
        }
    
    def preprocess_full_pipeline(self, file_path, use_smote=True):
        """
        Menjalankan full pipeline preprocessing.
        
        Parameters:
        -----------
        file_path : str
            Path ke file Excel.
        use_smote : bool
            Apakah menggunakan SMOTE atau tidak.
            
        Returns:
        --------
        dict
            Dictionary berisi data split, scaler, dan data original.
        """
        # Load data
        data = self.load_data(file_path)
        
        # Create class column
        data = self.create_class_column(data)
        
        # Remove irrelevant columns
        data_processed = self.remove_irrelevant_columns(data)
        self.data_processed = data_processed
        
        # Separate features and target
        X = data_processed.drop(columns=['Class'])
        y = data_processed['Class']
        
        # Split and scale
        split_data = self.split_data(X, y, use_smote=use_smote)
        
        return {
            'split_data': split_data,
            'data_processed': data_processed,
            'X_original': X,
            'y_original': y,
        }
