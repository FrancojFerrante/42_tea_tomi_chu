# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:18:10 2024

@author: Usach
"""

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix

# Cargar el dataframe desde un archivo de Excel
df = pd.read_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/IG_Analisis2_shuffle_decision_scores.xlsx")

# Umbrales típicos para cada modelo en sklearn
thresholds = {
    'Logistic Regression': 0.0,
    'Random Forest': 0.5,
    'SVM': 0.0  # Para decision_function en SVM
}

# Función para calcular métricas
def calculate_metrics(df, thresholds):
    models = df['Modelo'].unique()
    results = []

    for model in models:
        model_df = df[df['Modelo'] == model]
        iterations = model_df['Iteracion'].unique()

        for iteration in iterations:
            iter_df = model_df[model_df['Iteracion'] == iteration]
            y_true = iter_df['target']
            y_score = iter_df['Decision Score']
            
            # Aplicar el umbral específico para cada modelo
            threshold = thresholds[model]
            y_pred = (y_score > threshold).astype(int)

            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_score)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)

            # Calcular la matriz de confusión
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensibilidad = recall  # Sensibilidad es lo mismo que recall
            especificidad = tn / (tn + fp)  # Especificidad
            uar = (sensibilidad + especificidad) / 2  # Unweighted Average Recall (UAR)

            results.append({
                'Modelo': model,
                'Iteracion': iteration,
                'Accuracy': accuracy,
                'AUC': auc,
                'F1_mean': f1,
                'Recall_mean': recall,
                'Precision_mean': precision,
                'Sensibilidad_mean': sensibilidad,
                'Especificidad_mean': especificidad,
                'UAR_mean': uar
            })

    results_df = pd.DataFrame(results)

    # Calcular el promedio, el desvío estándar y el intervalo de confianza por iteración
    Z = 1.96
    n = len(results_df['Iteracion'].unique())  # Número de iteraciones

    summary_df = results_df.groupby('Modelo').agg(
        Accuracy_mean=('Accuracy', 'mean'),
        Accuracy_std=('Accuracy', 'std'),
        AUC_mean=('AUC', 'mean'),
        AUC_std=('AUC', 'std'),
        F1_mean_mean=('F1_mean', 'mean'),
        F1_mean_std=('F1_mean', 'std'),
        Recall_mean_mean=('Recall_mean', 'mean'),
        Recall_mean_std=('Recall_mean', 'std'),
        Precision_mean_mean=('Precision_mean', 'mean'),
        Precision_mean_std=('Precision_mean', 'std'),
        Sensibilidad_mean_mean=('Sensibilidad_mean', 'mean'),
        Sensibilidad_mean_std=('Sensibilidad_mean', 'std'),
        Especificidad_mean_mean=('Especificidad_mean', 'mean'),
        Especificidad_mean_std=('Especificidad_mean', 'std'),
        UAR_mean_mean=('UAR_mean', 'mean'),
        UAR_mean_std=('UAR_mean', 'std')
    ).reset_index()

    # Calcular los intervalos de confianza
    for metric in ['Accuracy', 'AUC', 'F1_mean', 'Recall_mean', 'Precision_mean', 'Sensibilidad_mean', 'Especificidad_mean', 'UAR_mean']:
        summary_df[f'{metric}_CI lower'] = summary_df[f'{metric}_mean'] - Z * (summary_df[f'{metric}_std'] / (n**0.5))
        summary_df[f'{metric}_CI upper'] = summary_df[f'{metric}_mean'] + Z * (summary_df[f'{metric}_std'] / (n**0.5))

    # Reordenar las columnas para que las métricas estén continuas
    columns = ['Modelo']
    for metric in ['Accuracy', 'AUC', 'F1_mean', 'Recall_mean', 'Precision_mean', 'Sensibilidad_mean', 'Especificidad_mean', 'UAR_mean']:
        columns += [f'{metric}_mean', f'{metric}_std', f'{metric}_CI lower', f'{metric}_CI upper']
    
    summary_df = summary_df[columns]

    return summary_df

# Ejemplo de uso
summary_df = calculate_metrics(df, thresholds)
print(summary_df)
summary_df.to_excel("C:/Franco/Doctorado/Laboratorio/42_tea_tomi_chu/resultados_machine_learning/IG_Analisis2_shuffle_decision_scores_from_scores.xlsx",index=False)
