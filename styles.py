# styles.py
import streamlit as st

def apply_custom_css():
    """
    Fungsi untuk menerapkan CSS custom di Streamlit.
    Bisa dipakai untuk mengganti font, warna, padding, dll.
    """
    st.markdown(
        """
        <style>
        /* Ganti font default */
        html, body, [class*="css"]  {
            font-family: 'Arial', sans-serif;
        }

        /* Judul besar */
        .stTitle {
            color: #1f77b4;
        }

        /* Tabel data */
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }

        /* Sidebar */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
