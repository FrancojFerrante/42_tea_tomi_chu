# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 18:34:17 2023

@author: Usach
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# Carga los datos en un DataFrame (asegúrate de tener el archivo de datos en el mismo directorioorioorio)

# Cargar datos
data_dcl = pd.read_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/BASE FINAL 2025_2_XP_renombrada.xlsx")

group_column = 'Grupo'
id_column = "ID"

# Reemplazar valores de Grupo
data_dcl[group_column] = data_dcl[group_column].map({'NT': 0, 'TEA': 1})

# Columnas a conservar inicialmente
columnas = [id_column,group_column,'IG_MLU_Words', 
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

# data_dcl = data_dcl[columnas]

# Columnas base del Análisis 1
analisis1 = ['MLU_Words', 'Riqueza_lexica', '1er_TotPron', '2da_TotPron', '3era_TotPron',
             '%_Pauses', '%_Fillers', '%_Reformulaciones', '%_monoWWR',
             'Rango_pitch', 'Rango_Jitter', 'Rango Shimmer', 'talking_intervals__speechrate']

# Columnas extra para análisis 3 y 4
extra_cols = ['palabras', '%_WWR', '%_Phonological_fragment']

# Columnas a eliminar para análisis 2 y 4
elim_cols = ['%_Reformulaciones', 'talking_intervals__speechrate']

# Función para crear listas completas para IG o IR
def get_columns(prefix, base_cols, extras=[], exclude=[]):
    return [f"{prefix}_{col}" for col in base_cols + extras if col not in exclude]

data_dcl[group_column] = np.random.permutation(data_dcl[group_column].values)

# Crear los DataFrames para cada análisis
dict_dfs = {
    # Análisis individuales
    # "IG_Analisis1": data_dcl[get_columns("IG", analisis1) + [group_column, id_column]],
    # "IR_Analisis1": data_dcl[get_columns("IR", analisis1) + [group_column, id_column]],

    "IG_Analisis2_shuffle": data_dcl[get_columns("IG", analisis1, exclude=elim_cols) + [group_column, id_column]],
    "IR_Analisis2_shuffle": data_dcl[get_columns("IR", analisis1, exclude=elim_cols) + [group_column, id_column]],

    # "IG_Analisis3": data_dcl[get_columns("IG", analisis1, extras=extra_cols) + [group_column, id_column]],
    # "IR_Analisis3": data_dcl[get_columns("IR", analisis1, extras=extra_cols) + [group_column, id_column]],

    # "IG_Analisis4": data_dcl[get_columns("IG", analisis1, extras=extra_cols, exclude=elim_cols) + [group_column, id_column]],
    # "IR_Analisis4": data_dcl[get_columns("IR", analisis1, extras=extra_cols, exclude=elim_cols) + [group_column, id_column]],

    # Análisis conjuntos IR_IG
    # "IR_IG_Analisis1": data_dcl[get_columns("IR", analisis1) + get_columns("IG", analisis1) + [group_column, id_column]],
    "IR_IG_Analisis2_shuffle": data_dcl[get_columns("IR", analisis1, exclude=elim_cols) + get_columns("IG", analisis1, exclude=elim_cols) + [group_column, id_column]],
    # "IR_IG_Analisis3": data_dcl[get_columns("IR", analisis1, extras=extra_cols) + get_columns("IG", analisis1, extras=extra_cols) + [group_column, id_column]],
    # "IR_IG_Analisis4": data_dcl[get_columns("IR", analisis1, extras=extra_cols, exclude=elim_cols) + get_columns("IG", analisis1, extras=extra_cols, exclude=elim_cols) + [group_column, id_column]],

}

# label_mapping = {'dcl': 1, 'no_dcl': 0}

results = []
models = [
    ('Logistic Regression', LogisticRegression(max_iter = 100000), {'C': [0.00001,0.0001, 0.001,0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200]}),
    ('SVM', SVC(max_iter = 100000), {'kernel': ['linear'], 'C': [0.00001,0.0001, 0.001,0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]})
    # ('XGBoost', XGBClassifier(), {'max_depth': [3, 5, 7], 'learning_rate': [0.00001,0.0001, 0.001,0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'n_estimators': [10, 100, 1000, 3000, 5000]})
    
    # ('Logistic Regression', LogisticRegression(), {'C': [0.00001,0.0001, ]}),
    # ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100]}),
    # ('SVM', SVC(), {'kernel': ['rbf', 'linear'], 'C': [0.00001,0.0001]}),
    # ('XGBoost', XGBClassifier(), {'max_depth': [3, 5], 'learning_rate': [0.00001,0.0001]})
]

# %% Grafico de barras

def calcular_percentil_25(columna):
    return np.percentile(columna, 25)

def calcular_percentil_75(columna):
    return np.percentile(columna, 75)


def grafico_barras_horizontales(df, x_label='Columnas', y_label='Promedio', title='Promedio y Desvío Estándar de Columnas', path=None):
    
    # Identificar las columnas numéricas
    df_columnas_numericas = df.select_dtypes(include=['number'])
    
    # Calcular promedio y desvío estándar de cada columna
    promedios = df_columnas_numericas.abs().mean()
    desvios = df_columnas_numericas.abs().std()
    
    # Aplica la función a cada columna
    q1 = df_columnas_numericas.apply(calcular_percentil_25)    #q3 = np.percentile(datos, 75)  # Tercer cuartil (Q3)
    q3 = df_columnas_numericas.apply(calcular_percentil_75)    #q3 = np.percentile(datos, 75)  # Tercer cuartil (Q3)

    iqr = q3 - q1
    
    # Ordenar por promedio de mayor a menor
    promedios_sorted = promedios.sort_values(ascending=False)
    
    # Crear el gráfico de barras horizontales
    fig, ax = plt.subplots()
    
    # Posiciones de las barras
    y_pos = np.arange(len(promedios_sorted))
    
    # Dibujar las barras horizontales
    bars = ax.barh(y_pos, promedios_sorted, xerr=desvios[promedios_sorted.index], align='center', alpha=0.7)
    
    # Etiquetas de las columnas en el eje Y
    ax.set_yticks(y_pos)
    ax.set_yticklabels(promedios_sorted.index)
    
    # Etiquetas y título
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Crear un DataFrame con los datos de variable, promedio y std
    df_result = pd.DataFrame({
        'variable': promedios_sorted.index,
        'promedio': promedios_sorted.values,
        'std': desvios[promedios_sorted.index].values,
        'iqr': iqr[promedios_sorted.index]
    })
    
    if path is not None:
        df_result.to_excel(path + ".xlsx",index=False)
        plt.savefig(path + '.png', bbox_inches='tight')
        
    # Mostrar el gráfico
    plt.show()



# %%
i_model = 0
for key, value in dict_dfs.items():
    # Divide los datos en características (variables independientes) y la variable objetivo
    X = value.drop([group_column,id_column], axis=1)  # Características
    y = value[group_column]  # Variable objetivo
    # Crea una instancia del codificador de etiquetas
    # label_encoder = LabelEncoder()
    i_model+=1
    # Convierte las etiquetas categóricas a valores numéricos
    # y = label_encoder.fit_transform(y)
    
    # y = y.map(label_mapping)

    results = []
    
    # Aplica la normalización Min-Max a los datos de entrenamiento
    scaler = MinMaxScaler()
    imputer = KNNImputer(n_neighbors=3)

    n = 10
    list_all_results=[]
    decision_scores = []
    print(key)
    
    df_feature_importance = pd.DataFrame(columns=["modelo"] + list(X.columns))


    for i_loop in range (0,n):
        print(i_loop)
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True)
        for name, model, params in models:
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
            grid_search = GridSearchCV(model, params, cv=inner_cv,n_jobs=-1)
        
            best_params_list = []
            best_model_list = []
            accuracy_list = []
            auc_list = []
            f1_list = []
            recall_list = []
            precision_list = []
            sens_list = []
            spec_list = []
            uar_list = []
            
            i_outer_loop = 0
            for train_index, test_index in outer_cv.split(X, y):
                i_outer_loop+=1
                X_train, X_val = X.iloc[train_index], X.iloc[test_index]
                y_train, y_val = y[train_index], y[test_index]
        
                X_train = scaler.fit_transform(X_train)
            
                # Aplica la misma transformación a los datos de prueba
                X_val = scaler.transform(X_val)
                
                # Ajustar y transformar los datos de entrenamiento
                X_train = imputer.fit_transform(X_train)
                
                # Transformar los datos de prueba utilizando el imputador ya ajustado
                X_val = imputer.transform(X_val)

                grid_search.fit(X_train, y_train)
        
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
        
                best_model_list.append(best_model)
                best_params_list.append(best_params)
        
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_val)
                
                if (name == "Random Forest") or (name == "XGBoost"):
                    decision_scores_actual = best_model.predict_proba(X_val)[:,1]
                else:
                    decision_scores_actual = best_model.decision_function(X_val)
                    
                # Feature importance
                if (name == "Logistic Regression"):
                    feature_importance = best_model.coef_[0]
                    feature_importance_named = dict(zip(X.columns, feature_importance))
                elif (name == "Random Forest"):
                    feature_importance = best_model.feature_importances_
                    feature_importance_named = dict(zip(X.columns, feature_importance))
                elif (name == "SVM"):
                    # Obtener los vectores de soporte
                    support_vectors = best_model.support_vectors_
                    # Puedes analizar los coeficientes para estimar la importancia relativa
                    coeficients = best_model.coef_[0]
                    feature_importance_named = dict(zip(X.columns, coeficients))
                elif (name == "XGBoost"):
                    feature_importance = best_model.feature_importances_
                    feature_importance_named = dict(zip(X.columns, feature_importance))
        
                # Normalizar las importancias de características
                features_scaler = MinMaxScaler()
                feature_importance_normalized = features_scaler.fit_transform(np.array(list(feature_importance_named.values())).reshape(-1, 1)).flatten()
                
                df_feature_importance.loc[len(df_feature_importance)] = [name]+list(feature_importance_normalized)
        
                accuracy = accuracy_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                
                # Calcula sensibilidad, especificidad y UAR
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                sens = tp / (tp + fn)
                spec = tn / (tn + fp)
                uar = (sens + spec) / 2
    
                accuracy_list.append(accuracy)
                auc_list.append(auc)
                f1_list.append(f1)
                recall_list.append(recall)
                precision_list.append(precision)
                sens_list.append(sens)
                spec_list.append(spec)
                uar_list.append(uar)
    
                inner_result = {
                    'Iteracion': i_loop,
                    'Modelo': name,
                    'Mejores hiperparámetros': best_params,
                    'Accuracy': accuracy,
                    'AUC': auc,
                    'F1': f1,
                    'Recall': recall,
                    'Precision': precision,
                    'Sensibilidad': sens,
                    'Especificidad': spec,
                    'UAR': uar
                }
                list_all_results.append(inner_result)
                
                # Guardar los decision scores
                for i in range(len(decision_scores_actual)):
                    decision_scores.append({
                        'Iteracion': i_loop,
                        'i_outer_loop': i_outer_loop,
                        'Modelo': name,
                        'Mejores hiperparámetros': best_params,
                        'Decision Score': decision_scores_actual[i],
                        'target': y_val[test_index[i]],
                        'PatientId': value.iloc[test_index[i]][id_column]  # Assuming 'value' is your DataFrame

                    })
    
    df_results_inner = pd.DataFrame(list_all_results)

    # Guarda el DataFrame en un archivo Excel
    df_results_inner.to_excel('C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning//' + key + '_all.xlsx', index=False)
    df_feature_importance.to_excel('C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/feature_importance/' + key + '_all.xlsx', index=False)

    for name, model, params in models:
        df_feature_importance_modelo = df_feature_importance[df_feature_importance["modelo"] == name]

        # Identificar las columnas numéricas
        columnas_numericas = df_feature_importance_modelo.select_dtypes(include=['number'])
        
        # Calcular el promedio de los valores absolutos de las columnas numéricas
        esto = columnas_numericas.abs().mean()        
        
        grafico_barras_horizontales(df_feature_importance_modelo,title=key + "_" + name + '_all',path="C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/feature_importance/" + key + "_" + name + '_promedio')
    df_promedio = df_results_inner.groupby('Modelo').agg(['mean', 'std'])
    df_promedio.columns = [f'{col}_{stat}' for col, stat in df_promedio.columns]
    df_promedio = df_promedio.reset_index()
    
    df_mejores = pd.DataFrame()
    # Expandir los diccionarios en columnas separadas
    df = pd.concat([df_results_inner.drop('Mejores hiperparámetros', axis=1), df_results_inner['Mejores hiperparámetros'].apply(pd.Series)], axis=1)
    
    # Agrupar por "Modelo" y calcular el promedio de los valores de cada clave
    df_mean = df.drop(['Iteracion', 'Accuracy','AUC', 'F1','Recall', 'Sensibilidad', 'Especificidad', 'UAR'], axis=1).groupby('Modelo').mean().reset_index()
    
    # Obtener las columnas (excepto 'Modelo')
    columnas = df_mean.columns.drop('Modelo')
    
    # Crear la nueva columna 'mejores hiperparámetros' sin claves NaN
    df_mean['mejores hiperparámetros'] = df_mean[columnas].apply(lambda row: {k: v for k, v in row.dropna().items()}, axis=1)
    
    # Seleccionar solo las columnas 'Modelo' y 'mejores hiperparámetros'
    df_resultado = df_mean[['Modelo', 'mejores hiperparámetros']]
    
    df_promedio = df_promedio.merge(df_resultado)
    # Calcula la moda de los hiperparámetros
    def calculate_mode(hyperparameters):
        flattened_params = [tuple(sorted(params.items())) for params in hyperparameters]
        counts = {param: flattened_params.count(param) for param in flattened_params}
        max_count = max(counts.values())
        modes = [param for param, count in counts.items() if count == max_count]
        return ', '.join([str(dict(param)) for param in modes])
    
    df_promedio['Moda hiperparámetros'] = df_results_inner.groupby('Modelo')['Mejores hiperparámetros'].apply(calculate_mode).values
    df_promedio.rename(columns = {'mejores hiperparámetros':'Promedio hiperparámetros'}, inplace = True)
    
    # Guardar los scores de decisión en un archivo Excel
    df_decision_scores = pd.DataFrame(decision_scores)
    
    df_promedio.to_excel('C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/' + key + '_promedio.xlsx', index=False)
    df_decision_scores.to_excel('C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/' + key + '_decision_scores.xlsx', index=False)

