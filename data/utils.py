"""
Utility functions untuk data management.
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def ensure_datasets_folder():
    """Create datasets folder jika belum ada."""
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    return datasets_dir

def get_sample_dataset_path():
    """Get path ke sample dataset."""
    ensure_datasets_folder()
    return Path("datasets")

def ensure_outputs_folder():
    """Create outputs folder untuk menyimpan hasil export jika belum ada."""
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    return outputs_dir

def export_labeled_dataset(data, filename=None, use_timestamp=True):
    """
    Export dataset dengan label/class ke folder outputs.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame yang sudah memiliki kolom 'Class'.
    filename : str, optional
        Nama file output. Jika None, akan generate nama otomatis.
    use_timestamp : bool
        Jika True, tambahkan timestamp di nama file.
    
    Returns:
    --------
    str
        Path ke file yang sudah di-export.
    """
    outputs_dir = ensure_outputs_folder()
    
    if filename is None:
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ipm_classified_{timestamp}.xlsx"
        else:
            filename = "ipm_classified.xlsx"
    
    filepath = outputs_dir / filename
    
    # Export ke Excel
    data.to_excel(filepath, sheet_name='Classified Data', index=False)
    
    return str(filepath)

def export_balanced_dataset(X_train, y_train, X_test=None, y_test=None, filename=None):
    """
    Export dataset balanced (after SMOTE) dengan pemisahan train/test.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame, optional
        Testing features.
    y_test : pd.Series, optional
        Testing target.
    filename : str, optional
        Nama file base untuk output.
    
    Returns:
    --------
    dict
        Dictionary berisi paths dari file-file yang di-export.
    """
    outputs_dir = ensure_outputs_folder()
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"balanced_data_{timestamp}"
    else:
        base_filename = filename.replace('.xlsx', '')
    
    paths = {}
    
    # Export training set
    train_data = X_train.copy()
    train_data['Class'] = y_train.values
    train_filepath = outputs_dir / f"{base_filename}_train.xlsx"
    train_data.to_excel(train_filepath, sheet_name='Training Data', index=False)
    paths['train'] = str(train_filepath)
    
    # Export testing set jika tersedia
    if X_test is not None and y_test is not None:
        test_data = X_test.copy()
        test_data['Class'] = y_test.values
        test_filepath = outputs_dir / f"{base_filename}_test.xlsx"
        test_data.to_excel(test_filepath, sheet_name='Testing Data', index=False)
        paths['test'] = str(test_filepath)
    
    return paths

def export_original_with_predictions(X_data, y_original, y_predicted, filename=None):
    """
    Export dataset original dengan kolom original class dan predicted class.
    
    Parameters:
    -----------
    X_data : pd.DataFrame
        Features dataset.
    y_original : pd.Series
        Original target values.
    y_predicted : pd.Series
        Predicted target values.
    filename : str, optional
        Nama file output.
    
    Returns:
    --------
    str
        Path ke file yang di-export.
    """
    outputs_dir = ensure_outputs_folder()
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.xlsx"
    
    result_data = X_data.copy()
    result_data['Actual_Class'] = y_original.values
    result_data['Predicted_Class'] = y_predicted
    
    # Add match column
    result_data['Correct'] = result_data['Actual_Class'] == result_data['Predicted_Class']
    
    filepath = outputs_dir / filename
    result_data.to_excel(filepath, sheet_name='Predictions', index=False)
    
    return str(filepath)

def get_export_summary(outputs_dir_path=None):
    """
    Get list of semua file yang sudah di-export.
    
    Parameters:
    -----------
    outputs_dir_path : str, optional
        Path ke outputs folder.
    
    Returns:
    --------
    list
        List of file info.
    """
    if outputs_dir_path is None:
        outputs_dir = ensure_outputs_folder()
    else:
        outputs_dir = Path(outputs_dir_path)
    
    if not outputs_dir.exists():
        return []
    
    files_info = []
    for file in outputs_dir.glob('*.xlsx'):
        file_stat = file.stat()
        files_info.append({
            'filename': file.name,
            'size': f"{file_stat.st_size / 1024:.2f} KB",
            'created': datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            'path': str(file)
        })
    
    return sorted(files_info, key=lambda x: x['created'], reverse=True)
