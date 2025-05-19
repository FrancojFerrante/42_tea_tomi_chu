# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:28:41 2023

@author: fferrante
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# %% Grafico de barras

def generar_lista_colores(inicial, final, top_n):
    # Convertir los colores de hexadecimal a valores RGB
    r1, g1, b1 = int(inicial[1:3], 16), int(inicial[3:5], 16), int(inicial[5:], 16)
    r2, g2, b2 = int(final[1:3], 16), int(final[3:5], 16), int(final[5:], 16)
    
    # Calcular el paso para la interpolación
    step_r = (r2 - r1) / (top_n - 1)
    step_g = (g2 - g1) / (top_n - 1)
    step_b = (b2 - b1) / (top_n - 1)
    
    # Generar la lista de colores
    lista_colores = []
    for i in range(top_n):
        r = int(r1 + step_r * i)
        g = int(g1 + step_g * i)
        b = int(b1 + step_b * i)
        color_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
        lista_colores.append(color_hex)
    
    return lista_colores

def grafico_barras_horizontales(df, col_group=None, x_label='Columnas', y_label='Promedio', title='Promedio y Desvío Estándar de Columnas', path=None, top_n=5,color_inicial='#C77CFF',color_final='#e9ccff'):
    # formato para los labels
    font_axis_labels = {'family': 'arial',
        'color': 'black',
        'weight': 'bold',
        'size': 32,
    }

    if col_group is not None:
        df = df.groupby(col_group, as_index=False).mean()

    # Unir columnas con "_"
    df['combinada'] = df[col_group].apply(lambda row: ' '.join(row), axis=1)

    # Ordenar por promedio de mayor a menor
    df_columnas_numericas = df.sort_values(by='promedio', ascending=False)

    # Calcular promedio y desvío estándar de cada columna
    promedios = df_columnas_numericas["promedio"]
    desvios = df_columnas_numericas["iqr"]

    # Colores personalizados para las barras
    colores_personalizados = generar_lista_colores(color_inicial,color_final,top_n)

    # Tomar las primeras top_n características
    df_columnas_numericas_top = df_columnas_numericas.head(top_n)
    
    # Calcular promedio y desvío estándar de cada columna
    promedios_top_n = df_columnas_numericas_top["promedio"]
    desvios_top_n = df_columnas_numericas_top["iqr"]
    
    # Crear el gráfico de barras horizontales
    fig, ax = plt.subplots()

    # Posiciones de las barras
    y_pos = np.arange(len(promedios_top_n))
    y_pos = y_pos[::-1] 

    # Dibujar las barras horizontales
    bars = ax.barh(y_pos, promedios_top_n, xerr=desvios_top_n.values, align='center', alpha=0.7, color=colores_personalizados)
    # Etiquetas de las columnas en el eje Y
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_columnas_numericas_top["combinada"].values)

    # Etiquetas y título
    ax.set_xlabel(x_label, fontdict=font_axis_labels)
    ax.set_ylabel("", fontdict=font_axis_labels)
    ax.set_title("")
    plt.xticks(fontsize=20, fontname="arial")
    plt.yticks(fontsize=20, fontname="arial")

    # Crear un DataFrame con los datos de variable, promedio y std
    df_result = pd.DataFrame({
        'variable': df_columnas_numericas["combinada"].values,
        'promedio': promedios.values,
        'iqr': desvios.values
    })

    if path is not None:
        df_result.to_excel(path + ".xlsx", index=False)
        plt.savefig(path + '.png', bbox_inches='tight', dpi=600)

    # Mostrar el gráfico
    plt.show()
    

# %%
# Specify the folder path containing the .xlsx files
folder_path = 'C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/feature_importance'

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Filter out only the .xlsx files
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

# Iterate through each .xlsx file and read it using pandas
for xlsx_file in xlsx_files:
    
        if ("promedio" in xlsx_file):
        
            file_path = os.path.join(folder_path, xlsx_file)
            df = pd.read_excel(file_path)  # Read the Excel file into a DataFrame
            # print(f"Contents of {xlsx_file}:")
            # print(df)  # Display the contents of the DataFrame
            # print("\n")
            # df['variable'] = df['variable'].str.replace('letra_p_', 'p_')
            
            # for i_row, row in df.iterrows():
                # df.loc[i_row,"variable"] = df.loc[i_row,"variable"].replace("sa_num_phon", "Phonemes")
                # df.loc[i_row,"variable"] = df.loc[i_row,"variable"].replace("log_frq", "Frequency")
                # df.loc[i_row,"variable"] = df.loc[i_row,"variable"].replace("sa_NP","Neighbors")
                # df.loc[i_row,"variable"] = df.loc[i_row,"variable"].replace("concreteness","Concreteness")
                # df.loc[i_row,"variable"] = df.loc[i_row,"variable"].replace("granularidad","Granularity")
            # df['variable'] = df['variable'].replace({"sa_num_phon": "Phonemes", "log_frq": "Frequency",\
            #                                        "sa_NP":"Neighbors",\
            #                                         "concreteness":"Concreteness",\
            #                                         "granularidad":"Granularity",\
            #                                         # "":"OSV"
            #                                         })
                
            # df[['tarea', 'feature', 'estadistico']] = df['variable'].str.split('_', 2, expand=True)
            
            # df["feature"] = None
            # for i_row, row in df.iterrows():
                # separado = row["variable"].split("_")
                # df.at[i_row,'feature'] = separado[1]
                # df.at[i_row,'feature_estadistico'] = separado[-1]
            
            # df[['feature', 'estadistico']] = df['feature_estadistico'].str.rsplit('_', 1, expand=True)
            
            # df['tarea'] = df['tarea'].replace({"p": "PhF", "super": "ThSF", "animales":"TaxSF"})
            # df['feature'] = df['feature'].replace({"Nphon": "Phonemes", "Lg10WF": "Frequency",\
            #                                        "phon_neigh":"Neighbors",\
            #                                         "concreteness":"Concreteness",\
            #                                         "granularidad":"Granularity",\
            #                                         "osv":"OSV",
            #                                         "AoA_Kup":"Age of adquisition",
            #                                         "Nsyll":"Syllables",
            #                                         "sem_neigh_den":"Semantic neighborhood"})
        
            # df.loc[df['variable'] == 'words_lemma_osv', 'feature'] = 'osv'
            # df.loc[df['variable'] == 'TOTAL', 'feature'] = 'TOTAL'

            # n_tarea = len(np.unique(df["tarea"].values))
            # n_feature = len(np.unique(df["feature"].values))
            # Generar el producto cartesiano de las dos columnas
            # combinations = list(product(df['tarea'], df['feature']))
            
            # Contar la cantidad de combinaciones
            # n_tarea_feature = len(combinations)
            n_tarea_feature = 30
            # grafico_barras_horizontales(df,col_group=["tarea"],x_label="Feature importance",y_label="Task",title=file_path.split("\\")[-1].split(".")[0],\
            #                             path=file_path.split("\\")[0] + "//colapsada//" + file_path.split("\\")[-1].split(".")[0] + "_task", top_n=n_tarea)
            # grafico_barras_horizontales(df,col_group=["feature"],x_label="Feature importance",y_label="Feature",title=file_path.split("\\")[-1].split(".")[0],\
            #                             path=file_path.split("\\")[0] + "//colapsada//" + file_path.split("\\")[-1].split(".")[0] + "_feature", top_n=n_feature)
            grafico_barras_horizontales(df,col_group=["variable"],x_label="Importance",y_label="Feature",title=file_path.split("\\")[-1].split(".")[0],\
                                        path=file_path.split("\\")[0] + "//colapsada//" + file_path.split("\\")[-1].split(".")[0] + "_feature_task", top_n=n_tarea_feature)
            
            