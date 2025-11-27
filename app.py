"""
Main Streamlit Application untuk Analisis dan Klasifikasi IPM - Clean Version.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from data.data_loader import DataLoader
from models.model_trainer import ModelTrainer
from visualization.plots import DataVisualizer
from config import IPM_LABELS
from styles import apply_custom_css

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="IPM Classification Analysis",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# ==================== Helper Functions ====================
def get_available_datasets():
    """Get list of available datasets dari folder datasets/."""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        datasets_dir.mkdir(exist_ok=True)
        return []
    
    files = list(datasets_dir.glob("*.xlsx")) + list(datasets_dir.glob("*.xls"))
    return sorted([f.name for f in files])

def load_dataset_from_path(file_path):
    """Load dataset dari path."""
    return file_path

def calculate_imbalance_ratio(y_data):
    """
    Calculate imbalance ratio and identify minority/majority classes.
    """
    value_counts = y_data.value_counts()
    majority_count = value_counts.max()
    minority_count = value_counts.min()
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
    return imbalance_ratio, minority_count, majority_count

def get_imbalance_warning_and_recommendation(imbalance_ratio):
    """
    Generate warning and recommendation based on imbalance ratio.
    """
    if imbalance_ratio >= 4:
        severity = "SEVERE"
        color = "#E5243B"
        recommendation = "SMOTE strongly recommended. Consider ADASYN or class weights as alternatives."
    elif imbalance_ratio >= 2.5:
        severity = "HIGH"
        color = "#DDA63B"
        recommendation = "SMOTE recommended for better minority class representation."
    elif imbalance_ratio >= 1.5:
        severity = "MODERATE"
        color = "#F39C12"
        recommendation = "SMOTE can be beneficial. Also consider stratified cross-validation."
    else:
        severity = "BALANCED"
        color = "#4C9F38"
        recommendation = "Dataset is reasonably balanced. Standard training approaches work well."
    
    return severity, color, recommendation, imbalance_ratio

# ==================== Sidebar ====================
with st.sidebar:
    st.title("IPM Analytics")
    st.caption("Machine Learning Classification System")
    st.divider()
    
    # Data source selection
    st.subheader("Data Source")
    
    available_datasets = get_available_datasets()
    uploaded_file = None
    
    if available_datasets:
        data_source = st.radio(
            "Select data source",
            ["Upload File", "Use Local Dataset"],
            label_visibility="collapsed"
        )
        
        if data_source == "Use Local Dataset":
            selected_dataset = st.selectbox(
                "Choose dataset",
                available_datasets,
                label_visibility="collapsed"
            )
            dataset_path = Path("datasets") / selected_dataset
            uploaded_file = load_dataset_from_path(dataset_path)
            st.success(f"Loaded: {selected_dataset}")
        else:
            uploaded_file = st.file_uploader(
                "Upload Excel file",
                type=['xlsx', 'xls'],
                label_visibility="collapsed"
            )
    else:
        uploaded_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # Processing options
    st.subheader("Options")
    use_smote = st.checkbox("Apply SMOTE for balancing", value=True)
    
    model_option = st.selectbox(
        "Select model",
        ["SVM", "KNN", "Random Forest"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.subheader("Info")
    st.info("""
    This application analyzes IPM (Human Development Index) classification using machine learning with handling for imbalanced data through SMOTE techniques.
    """)

# ==================== Main Content ====================
st.title("IPM Classification Analysis")
st.caption("Powered by Machine Learning | Data Imbalance Handling with SMOTE")

if uploaded_file is None:
    st.warning("Please upload an Excel file or select a dataset from the sidebar to begin analysis")
    st.info("Expected file format: Excel with 'Dataset Imbalance' sheet containing IPM data")
    st.stop()

# ==================== Data Loading & Preprocessing ====================
@st.cache_data
def load_and_preprocess(file, use_smote_flag):
    """Load dan preprocess data."""
    loader = DataLoader()
    result = loader.preprocess_full_pipeline(file, use_smote=use_smote_flag)
    return result

try:
    result = load_and_preprocess(uploaded_file, use_smote)
    
    split_data = result['split_data']
    data_processed = result['data_processed']
    X_original = result['X_original']
    y_original = result['y_original']
    
    X_train_scaled = split_data['X_train']
    X_test_scaled = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    
except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.stop()

# ==================== Tabs ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Overview",
    "Imbalance Analysis",
    "Data Exploration",
    "Model Training",
    "Feature Analysis",
    "Model Comparison"
])

# ==================== Tab 1: Dataset ====================
with tab1:
    st.subheader("Dataset Structure")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(data_processed))
    with col2:
        st.metric("Features", X_original.shape[1])
    with col3:
        st.metric("Target Classes", data_processed['Class'].nunique())
    
    st.divider()
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(data_processed.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Statistics")
        st.dataframe(
            data_processed.describe().T.round(2),
            use_container_width=True
        )
    
    st.divider()
    st.subheader("Export Dataset")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("Export Original + Labels", use_container_width=True, key="export_labeled"):
            from data.utils import export_labeled_dataset
            filepath = export_labeled_dataset(data_processed)
            st.success(f"Dataset exported successfully!")
            st.caption(f"Location: outputs/{Path(filepath).name}")
            
            # Provide download button
            with open(filepath, 'rb') as f:
                st.download_button(
                    label="Download File",
                    data=f.read(),
                    file_name=Path(filepath).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col_export2:
        if st.button("Export Balanced Data (Train/Test)", use_container_width=True, key="export_balanced"):
            from data.utils import export_balanced_dataset
            paths = export_balanced_dataset(X_train_scaled, y_train, X_test_scaled, y_test)
            st.success("Balanced datasets exported successfully!")
            
            for data_type, filepath in paths.items():
                st.caption(f"{data_type.capitalize()}: outputs/{Path(filepath).name}")
    
    with col_export3:
        if st.button("View Export History", use_container_width=True, key="view_history"):
            from data.utils import get_export_summary
            export_files = get_export_summary()
            
            if export_files:
                st.subheader("Recent Exports")
                export_df = pd.DataFrame(export_files)
                st.dataframe(
                    export_df[['filename', 'size', 'created']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No exports found yet.")
    
    st.divider()
    st.subheader("Processing Pipeline")
    
    col_step1, col_step2, col_step3 = st.columns(3)
    
    with col_step1:
        st.markdown("""
        **Step 1: Classification**
        
        IPM binning into 4 classes:
        - Rendah (< 60)
        - Menengah (60-69.9)
        - Tinggi (69.9-79.9)
        - Sangat Tinggi (>= 79.9)
        """)
    
    with col_step2:
        status = "Enabled" if use_smote else "Disabled"
        st.markdown(f"""
        **Step 2: Data Balancing**
        
        SMOTE Status: **{status}**
        
        Handles class imbalance through oversampling
        """)
    
    with col_step3:
        st.markdown(f"""
        **Step 3: Train-Test Split**
        
        - Training: {len(y_train)} samples
        - Testing: {len(y_test)} samples
        - Scaler: StandardScaler
        """)

# ==================== Tab 2: Imbalance Analysis ====================
with tab2:
    st.subheader("Class Imbalance Analysis")
    
    imbalance_ratio, minority_count, majority_count = calculate_imbalance_ratio(y_original)
    severity, warning_color, recommendation, ratio_val = get_imbalance_warning_and_recommendation(imbalance_ratio)
    
    # Imbalance Warning Box
    col_warning1, col_warning2, col_warning3, col_warning4 = st.columns(4)
    
    with col_warning1:
        st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
    
    with col_warning2:
        st.metric("Majority Class", majority_count)
    
    with col_warning3:
        st.metric("Minority Class", minority_count)
    
    with col_warning4:
        st.metric("Severity", severity)
    
    st.divider()
    
    # Warning Alert
    if severity == "SEVERE":
        st.error(f"""
        **SEVERE IMBALANCE DETECTED**
        
        Imbalance Ratio: {imbalance_ratio:.2f}:1
        
        {recommendation}
        """)
    elif severity == "HIGH":
        st.warning(f"""
        **HIGH IMBALANCE DETECTED**
        
        Imbalance Ratio: {imbalance_ratio:.2f}:1
        
        {recommendation}
        """)
    elif severity == "MODERATE":
        st.info(f"""
        **MODERATE IMBALANCE**
        
        Imbalance Ratio: {imbalance_ratio:.2f}:1
        
        {recommendation}
        """)
    else:
        st.success(f"""
        **BALANCED DATASET**
        
        Imbalance Ratio: {imbalance_ratio:.2f}:1
        
        {recommendation}
        """)
    
    st.divider()
    
    # Distribution Comparison
    st.subheader("Distribution: Before vs After SMOTE")
    fig_imbalance = DataVisualizer.plot_imbalance_comparison(
        y_original,
        y_train,
        title="Class Distribution Comparison"
    )
    st.pyplot(fig_imbalance, use_container_width=True)
    
    st.divider()
    
    # Class Distribution Ratio
    st.subheader("Class Distribution Ratio")
    fig_ratio = DataVisualizer.plot_imbalance_ratio(y_original)
    st.pyplot(fig_ratio, use_container_width=True)
    
    st.divider()
    
    # SMOTE Impact
    st.subheader("SMOTE Impact Analysis")
    
    original_counts = y_original.value_counts().sort_index()
    train_counts = y_train.value_counts().sort_index()
    
    impact_data = []
    for class_label in original_counts.index:
        original = original_counts.get(class_label, 0)
        after_smote = train_counts.get(class_label, 0)
        increase = after_smote - original
        increase_pct = (increase / original * 100) if original > 0 else 0
        
        impact_data.append({
            'Class': class_label,
            'Original Count': original,
            'After SMOTE': after_smote,
            'Increase': increase,
            'Increase %': f"{increase_pct:.1f}%"
        })
    
    impact_df = pd.DataFrame(impact_data)
    st.dataframe(impact_df, use_container_width=True)

# ==================== Tab 3: Data Exploration ====================
with tab3:
    st.subheader("Data Visualization")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.markdown("**Original Distribution (Imbalanced)**")
        fig_original = DataVisualizer.plot_class_distribution(
            y_original,
            title="Class Distribution - Original",
            colors=['#E5243B', '#DDA63B', '#4C9F38', '#0A97D9']
        )
        st.pyplot(fig_original, use_container_width=True)
    
    with col_dist2:
        title_text = "Class Distribution - After SMOTE" if use_smote else "Class Distribution - No Sampling"
        st.markdown(f"**After Processing**")
        fig_processed = DataVisualizer.plot_class_distribution(
            y_train,
            title=title_text,
            colors=['#E5243B', '#DDA63B', '#4C9F38', '#0A97D9']
        )
        st.pyplot(fig_processed, use_container_width=True)
    
    st.divider()
    
    st.markdown("**Feature Correlation Heatmap**")
    fig_corr = DataVisualizer.plot_correlation_heatmap(
        X_original,
        title="Feature Correlations"
    )
    st.pyplot(fig_corr, use_container_width=True)

# ==================== Tab 4: Model Training ====================
with tab4:
    st.subheader(f"Training: {model_option}")
    
    st.info(f"Selected Model: **{model_option}**")
    
    trainer = ModelTrainer()
    
    with st.spinner(f"Training {model_option}..."):
        if model_option == "SVM":
            result_train = trainer.train_svm(X_train_scaled, y_train)
        elif model_option == "KNN":
            result_train = trainer.train_knn(X_train_scaled, y_train)
        else:
            result_train = trainer.train_random_forest(X_train_scaled, y_train)
        
        best_model = result_train['model']
        best_params = result_train['params']
    
    eval_result = trainer.evaluate_model(best_model, X_test_scaled, y_test)
    
    st.subheader("Evaluation Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Accuracy", f"{eval_result['accuracy']:.2%}")
    
    with col_m2:
        roc_val = eval_result['roc_auc'] if eval_result['roc_auc'] else 0
        st.metric("ROC AUC", f"{roc_val:.2%}")
    
    with col_m3:
        st.metric("Test Set Size", len(y_test))
    
    with col_m4:
        st.metric("Train Set Size", len(y_train))
    
    st.divider()
    
    st.subheader("Per-Class Performance Metrics")
    
    col_precision, col_recall, col_f1 = st.columns(3)
    
    with col_precision:
        st.markdown("**Precision per Class**")
        fig_precision = DataVisualizer.plot_per_class_metrics(
            eval_result['classification_report'],
            metric='precision',
            labels=IPM_LABELS
        )
        st.pyplot(fig_precision, use_container_width=True)
    
    with col_recall:
        st.markdown("**Recall per Class**")
        fig_recall = DataVisualizer.plot_per_class_metrics(
            eval_result['classification_report'],
            metric='recall',
            labels=IPM_LABELS
        )
        st.pyplot(fig_recall, use_container_width=True)
    
    with col_f1:
        st.markdown("**F1-Score per Class**")
        fig_f1 = DataVisualizer.plot_per_class_metrics(
            eval_result['classification_report'],
            metric='f1-score',
            labels=IPM_LABELS
        )
        st.pyplot(fig_f1, use_container_width=True)
    
    st.divider()
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(eval_result['classification_report']).T
        st.dataframe(report_df.round(3), use_container_width=True)
    
    with col_eval2:
        st.markdown("**Confusion Matrix**")
        fig_cm = DataVisualizer.plot_confusion_matrix(
            eval_result['confusion_matrix'],
            labels=IPM_LABELS,
            title=f"Confusion Matrix"
        )
        st.pyplot(fig_cm, use_container_width=True)
    
    st.divider()
    st.markdown("**Hyperparameters**")
    params_df = pd.DataFrame([best_params])
    st.dataframe(params_df.T, use_container_width=True)

# ==================== Tab 5: Feature Importance ====================
with tab5:
    st.subheader("Feature Importance Analysis")
    
    trainer = ModelTrainer()
    if model_option == "SVM":
        result_train = trainer.train_svm(X_train_scaled, y_train)
    elif model_option == "KNN":
        result_train = trainer.train_knn(X_train_scaled, y_train)
    else:
        result_train = trainer.train_random_forest(X_train_scaled, y_train)
    
    best_model = result_train['model']
    feature_importance = trainer.get_feature_importance(
        best_model,
        X_train_scaled.columns.tolist()
    )
    
    if feature_importance is not None:
        top_n = st.slider("Number of features to display", 5, len(feature_importance), 6)
        
        fig_imp = DataVisualizer.plot_feature_importance(
            feature_importance,
            top_n=top_n,
            title=f"Top {top_n} Important Features",
            colors=['#0A97D9']
        )
        st.pyplot(fig_imp, use_container_width=True)
        
        st.divider()
        st.markdown("**Feature Importance Values**")
        st.dataframe(feature_importance.head(15), use_container_width=True)
    else:
        st.warning("This model does not support feature importance analysis.")

# ==================== Tab 6: Model Comparison ====================
with tab6:
    st.subheader("Model Performance Comparison")
    
    st.info("Comparing all three models (SVM, KNN, Random Forest) on the same dataset")
    
    trainer = ModelTrainer()
    comparison_results = []
    models_to_compare = ["SVM", "KNN", "Random Forest"]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(models_to_compare):
        status_text.text(f"Training {model_name}... ({idx+1}/{len(models_to_compare)})")
        
        if model_name == "SVM":
            result = trainer.train_svm(X_train_scaled, y_train)
        elif model_name == "KNN":
            result = trainer.train_knn(X_train_scaled, y_train)
        else:
            result = trainer.train_random_forest(X_train_scaled, y_train)
        
        eval_res = trainer.evaluate_model(result['model'], X_test_scaled, y_test)
        
        comparison_results.append({
            'Model': model_name,
            'Accuracy': eval_res['accuracy'],
            'ROC AUC': eval_res['roc_auc'] if eval_res['roc_auc'] else 0,
            'Precision': eval_res['classification_report']['weighted avg']['precision'],
            'Recall': eval_res['classification_report']['weighted avg']['recall'],
            'F1-Score': eval_res['classification_report']['weighted avg']['f1-score']
        })
        
        progress_bar.progress((idx + 1) / len(models_to_compare))
    
    status_text.empty()
    progress_bar.empty()
    
    comparison_df = pd.DataFrame(comparison_results)
    
    col_comp1, col_comp2 = st.columns([1, 1.2])
    
    with col_comp1:
        st.markdown("**Comparison Table**")
        st.dataframe(comparison_df.round(4), use_container_width=True)
    
    with col_comp2:
        st.markdown("**Performance Chart**")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x_pos = np.arange(len(comparison_df['Model']))
        width = 0.2
        
        colors_bars = ['#0A97D9', '#4C9F38', '#DDA63B']
        
        ax.bar(x_pos - width, comparison_df['Accuracy'], width, label='Accuracy', color=colors_bars[0], alpha=0.8)
        ax.bar(x_pos, comparison_df['ROC AUC'], width, label='ROC AUC', color=colors_bars[1], alpha=0.8)
        ax.bar(x_pos + width, comparison_df['F1-Score'], width, label='F1-Score', color=colors_bars[2], alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=11, fontweight='600')
        ax.set_title('Model Metrics Comparison', fontsize=13, fontweight='700', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comparison_df['Model'], fontsize=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

st.divider()
st.caption("IPM Classification Analysis System | Built with Streamlit & Scikit-learn")
