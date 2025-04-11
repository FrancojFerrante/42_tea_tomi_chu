# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 12:48:34 2025

@author: Usach
"""

import pandas as pd
import re

df_clasi = pd.read_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/solo_columnas_clasificador.xlsx")

df_clasi.columns = df_clasi.columns.str.replace('__', '_', regex=False)


df_clasi.columns = [
    re.sub(r'^(IG|IR)', r'\1__nada__', col)
    for col in df_clasi.columns
]

df_clasi["Grupo"] = df_clasi["Grupo"].replace({"NT": 0, "TEA": 1})

df_clasi.to_csv("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/solo_columnas_clasificador_formateadas.csv",index=False)
