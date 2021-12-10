# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:55:01 2021

@author: onero
"""

mkdir -p ~/.streamlit/
echo "[general]
email = \"email@com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml