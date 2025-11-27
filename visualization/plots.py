"""
Module untuk visualisasi dan plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import SDGS_COLORS, SDGS_CLASS_COLORS, SDGS_PLOT_COLORS


class DataVisualizer:
    """Class untuk visualisasi data dengan tema SDGs."""
    
    @staticmethod
    def plot_class_distribution(data, title="Distribusi Kelas", colors=None):
        """
        Plot distribusi kelas dengan SDGs color scheme.
        
        Parameters:
        -----------
        data : pd.Series
            Series berisi kelas.
        title : str
            Judul plot.
        colors : list, optional
            Custom color palette. Jika None, gunakan SDGS_CLASS_COLORS.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object.
        """
        if colors is None:
            colors = SDGS_CLASS_COLORS
        
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
        value_counts = data.value_counts().sort_index()
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors[:len(value_counts)], alpha=0.8, edgecolor='#2d3748', linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='#0A97D9', pad=20)
        ax.set_xlabel("Kelas", fontsize=12, fontweight='600', color='#2d3748')
        ax.set_ylabel("Jumlah Sampel", fontsize=12, fontweight='600', color='#2d3748')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, fontsize=10)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='600', color='#2d3748')
        
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='#e0e0e0')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(data, title="Korelasi Antar Fitur"):
        """
        Plot heatmap korelasi dengan SDGs color scheme.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame berisi fitur numerik.
        title : str
            Judul plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        corr_matrix = data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f",
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'Korelasi'},
                    vmin=-1, vmax=1, center=0)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='#0A97D9', pad=20)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(conf_matrix, labels, title="Confusion Matrix"):
        """
        Plot confusion matrix dengan SDGs styling.
        
        Parameters:
        -----------
        conf_matrix : np.ndarray
            Confusion matrix.
        labels : list
            Nama kelas.
        title : str
            Judul plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object.
        """
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Jumlah Prediksi'}, linewidths=1.5, linecolor='white')
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='#0A97D9', pad=20)
        ax.set_xlabel("Prediksi", fontsize=12, fontweight='600', color='#2d3748')
        ax.set_ylabel("Actual", fontsize=12, fontweight='600', color='#2d3748')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_df, top_n=10, title="Feature Importance", colors=None):
        """
        Plot feature importance dengan SDGs styling.
        
        Parameters:
        -----------
        importance_df : pd.DataFrame
            DataFrame berisi feature dan importance.
        top_n : int
            Jumlah top features untuk ditampilkan.
        title : str
            Judul plot.
        colors : list, optional
            Custom color palette. Jika None, gunakan SDGS_PLOT_COLORS.
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object.
        """
        if colors is None:
            colors = [SDGS_COLORS['primary']]
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        top_features = importance_df.head(top_n).sort_values('Importance')
        
        bars = ax.barh(range(len(top_features)), top_features['Importance'].values, 
                       color=colors[0] if isinstance(colors, list) else colors, 
                       alpha=0.8, edgecolor='#2d3748', linewidth=1.5)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold', color='#0A97D9', pad=20)
        ax.set_xlabel("Importance Score", fontsize=12, fontweight='600', color='#2d3748')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features['Importance'].values)):
            ax.text(value, bar.get_y() + bar.get_height()/2.,
                    f' {value:.4f}',
                    ha='left', va='center', fontsize=9, fontweight='600', color='#2d3748')
        
        ax.grid(axis='x', alpha=0.3, linestyle='--', color='#e0e0e0')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_imbalance_comparison(y_original, y_train, title="Class Distribution Comparison"):
        """
        Add new visualization comparing original vs SMOTE-balanced distribution.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        
        original_counts = y_original.value_counts().sort_index()
        train_counts = y_train.value_counts().sort_index()
        
        colors = ['#E5243B', '#DDA63B', '#4C9F38', '#0A97D9']
        
        # Before SMOTE
        bars1 = axes[0].bar(range(len(original_counts)), original_counts.values, 
                            color=colors[:len(original_counts)], alpha=0.8, edgecolor='#2d3748', linewidth=1.5)
        axes[0].set_title("Before SMOTE (Original)", fontsize=12, fontweight='bold', color='#2d3748')
        axes[0].set_xlabel("Class", fontsize=11, fontweight='600')
        axes[0].set_ylabel("Sample Count", fontsize=11, fontweight='600')
        axes[0].set_xticks(range(len(original_counts)))
        axes[0].set_xticklabels(original_counts.index, fontsize=10)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].set_axisbelow(True)
        
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='600')
        
        # After SMOTE
        bars2 = axes[1].bar(range(len(train_counts)), train_counts.values,
                            color=colors[:len(train_counts)], alpha=0.8, edgecolor='#2d3748', linewidth=1.5)
        axes[1].set_title("After SMOTE (Balanced)", fontsize=12, fontweight='bold', color='#2d3748')
        axes[1].set_xlabel("Class", fontsize=11, fontweight='600')
        axes[1].set_ylabel("Sample Count", fontsize=11, fontweight='600')
        axes[1].set_xticks(range(len(train_counts)))
        axes[1].set_xticklabels(train_counts.index, fontsize=10)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].set_axisbelow(True)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='600')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', color='#0A97D9', y=1.02)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_per_class_metrics(classification_report, metric='precision', labels=None):
        """
        Add visualization for per-class metrics (precision, recall, f1).
        """
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        metrics_data = {}
        for class_label in [key for key in classification_report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]:
            try:
                metrics_data[str(class_label)] = classification_report[class_label][metric]
            except:
                pass
        
        if not metrics_data:
            return fig
        
        classes = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        colors_bars = ['#E5243B', '#DDA63B', '#4C9F38', '#0A97D9'][:len(classes)]
        
        bars = ax.bar(classes, values, color=colors_bars, alpha=0.8, edgecolor='#2d3748', linewidth=1.5)
        
        ax.set_title(f"{metric.capitalize()} per Class", fontsize=13, fontweight='bold', color='#0A97D9', pad=20)
        ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='600')
        ax.set_xlabel("Class", fontsize=11, fontweight='600')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='600')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_imbalance_ratio(y_original, labels=None):
        """
        Add visualization for imbalance ratio.
        """
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        
        value_counts = y_original.value_counts().sort_index()
        percentages = (value_counts / value_counts.sum() * 100).values
        
        colors_bars = ['#E5243B', '#DDA63B', '#4C9F38', '#0A97D9'][:len(value_counts)]
        
        bars = ax.bar(range(len(value_counts)), percentages, color=colors_bars, alpha=0.8, edgecolor='#2d3748', linewidth=1.5)
        
        ax.set_title("Class Distribution Ratio", fontsize=13, fontweight='bold', color='#0A97D9', pad=20)
        ax.set_ylabel("Percentage (%)", fontsize=11, fontweight='600')
        ax.set_xlabel("Class", fontsize=11, fontweight='600')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, fontsize=10)
        ax.set_ylim([0, max(percentages) * 1.2])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        for bar, pct, count in zip(bars, percentages, value_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{pct:.1f}%\n({int(count)})', ha='center', va='bottom', fontsize=9, fontweight='600')
        
        plt.tight_layout()
        return fig
