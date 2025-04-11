# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:24:10 2025

@author: Usach
"""

import pandas as pd

df = pd.read_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/BASE FINAL 2025_2_XP.xls")


# Diccionario con los reemplazos
replacements = {
    "IR MLU": "IR_MLU_Words",
    "IR#_mono-WWR": "IR_#_monoWWR",
    "IG_Riqueza lexica": "IG_Riqueza_lexica",  # O quizás debería ser "IG_Riqueza_lexica"?
    "IR Riqueza lexica": "IR_Riqueza_lexica",
    "IR%_Pauses": "IR_%_Pauses",
    "IR#_Pauses": "IR_#_Pauses",
    "IR%_Phonological_fragment": "IR_%_Phonological_fragment",
    "IR#_Phonological_fragment": "IR_#_Phonological_fragment",
    "IR#_WWR": "IR_#_WWR",
    "IR%_WWR": "IR_%_WWR",
    "IR%_Fillers": "IR_%_Fillers",
    "IR#_Fillers": "IR_#_Fillers",
    "IG_reformulaciones": "IG_Reformulaciones",
    "IR%_mono-WWR": "IR_%_monoWWR",
    "IG_1er_TotPron_IG": "IG_1er_TotPron",
    "IR_1er_TotPron_IR": "IR_1er_TotPron",
    "IG_2da_TotPron_IG": "IG_2da_TotPron",
    "IR_2da_TotPron_IR": "IR_2da_TotPron",
    "IG_3era_TotPron_IG": "IG_3era_TotPron",
    "IR_3era_TotPron_IR": "IR_3era_TotPron",
}

# Renombramos columnas
df.rename(columns=replacements, inplace=True)

df.to_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/BASE FINAL 2025_2_XP_renombrada.xlsx",index=False)

# Supongamos que ya tienes tu DataFrame llamado df
# Extraemos las columnas que comienzan con IG e IR
ig_cols = [col for col in df.columns if col.startswith('IG')]
ir_cols = [col for col in df.columns if col.startswith('IR')]

# Quitamos los prefijos para poder comparar "pares"
ig_suffixes = {col[2:] for col in ig_cols}
ir_suffixes = {col[2:] for col in ir_cols}

# Buscamos los que no tienen par
ig_without_pair = ig_suffixes - ir_suffixes
ir_without_pair = ir_suffixes - ig_suffixes

# Mostramos las columnas faltantes con prefijos restaurados
ig_missing = [f'IG{suffix}' for suffix in ig_without_pair]
ir_missing = [f'IR{suffix}' for suffix in ir_without_pair]


columnas = ['ID','Grupo','IG_MLU_Words', 
            'IG_Riqueza_lexica', 'IG_1er_TotPron', 'IG_2da_TotPron',
            'IG_3era_TotPron', 'IG_%_Pauses', 'IG_%_Fillers',
            'IG_%_Reformulaciones', 'IG_%_monoWWR',
            'IG_Rango_pitch', 'IG_Rango_Jitter',
            'IG_Rango Shimmer', 'IG_talking_intervals__speechrate',
            'IR_MLU_Words', 
            'IR_Riqueza_lexica', 'IR_1er_TotPron', 'IR_2da_TotPron',
            'IR_3era_TotPron', 'IR_%_Pauses', 'IR_%_Fillers',
            'IR_%_Reformulaciones', 'IR_%_monoWWR',
            'IR_Rango_pitch', 'IR_Rango_Jitter',
            'IR_Rango Shimmer', 'IR_talking_intervals__speechrate'
            ]

    
df = df[columnas]
df.to_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/solo_columnas_clasificador.xlsx",index=False)
