# -*- coding: utf-8 -*-
"""
Proyecto: Segmentación de usuarios de Netflix / consumo de series y películas
Algoritmo: K-Means
Autor: Proyecto Integrador - Minería de Datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# ============================================================
# 1. CARGA Y EXPLORACIÓN INICIAL DEL DATASET
# ============================================================

print("=" * 60)
print("1. CARGA Y EXPLORACIÓN DEL DATASET")
print("=" * 60)

# Cargamos el archivo
df = pd.read_csv('netflix_usuarios_raw.csv')

print("\nPrimeros 10 registros:")
print(df.head(10))

print("\nDimensiones del dataset:", df.shape)
print("\nColumnas y tipos de datos:")
print(df.dtypes)

print("\nEstadísticas descriptivas (variables numéricas):")
print(df.describe().round(2))

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nRegistros duplicados:", df.duplicated().sum())

# ----------------------------------------------------------
# Gráficas exploratorias
# ----------------------------------------------------------

# Gráfica 1: Distribución de horas mensuales
plt.figure(figsize=(10, 5))
sns.histplot(df['horas_mensuales'], bins=30, kde=True, color='#E50914')
plt.title('Distribución de horas de consumo mensual')
plt.xlabel('Horas mensuales')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('eda_horas_mensuales.png', dpi=120)
plt.close()
print("\n[Gráfica guardada] eda_horas_mensuales.png")

# Gráfica 2: Distribución de género preferido
plt.figure(figsize=(10, 5))
orden_generos = df['genero_preferido'].value_counts().index
sns.countplot(data=df, y='genero_preferido', order=orden_generos, palette='Reds_r')
plt.title('Géneros preferidos por los usuarios')
plt.xlabel('Cantidad de usuarios')
plt.ylabel('Género')
plt.tight_layout()
plt.savefig('eda_generos.png', dpi=120)
plt.close()
print("[Gráfica guardada] eda_generos.png")

# Gráfica 3: Horas mensuales según tipo de suscripción
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='tipo_suscripcion', y='horas_mensuales',
            order=['Basic', 'Standard', 'Premium'], palette='Reds')
plt.title('Horas de consumo según tipo de suscripción')
plt.xlabel('Tipo de suscripción')
plt.ylabel('Horas mensuales')
plt.tight_layout()
plt.savefig('eda_suscripcion_horas.png', dpi=120)
plt.close()
print("[Gráfica guardada] eda_suscripcion_horas.png")

# Variables que usaremos para el clustering (justificación más adelante)
# horas_mensuales, frecuencia_semanal, rating_promedio, porcentaje_series
# Son las que mejor reflejan el comportamiento de consumo

# ============================================================
# 2. LIMPIEZA Y PREPARACIÓN DE DATOS
# ============================================================

print("\n" + "=" * 60)
print("2. LIMPIEZA Y PREPARACIÓN DE DATOS")
print("=" * 60)

df_clean = df.copy()

# Eliminamos duplicados
antes = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Duplicados eliminados: {antes - len(df_clean)}")

# Tratamiento de nulos
# - rating_promedio: imputamos con la mediana (es numérica y no está muy sesgada)
# - dispositivo_principal: imputamos con la moda
mediana_rating = df_clean['rating_promedio'].median()
df_clean['rating_promedio'] = df_clean['rating_promedio'].fillna(mediana_rating)

moda_dispositivo = df_clean['dispositivo_principal'].mode()[0]
df_clean['dispositivo_principal'] = df_clean['dispositivo_principal'].fillna(moda_dispositivo)

print(f"Nulos en rating_promedio imputados con mediana ({mediana_rating})")
print(f"Nulos en dispositivo_principal imputados con moda ({moda_dispositivo})")
print("Nulos restantes:", df_clean.isnull().sum().sum())

# Valores atípicos en horas_mensuales
# Usamos el método IQR para identificarlos
Q1 = df_clean['horas_mensuales'].quantile(0.25)
Q3 = df_clean['horas_mensuales'].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['horas_mensuales'] < limite_inf) |
                    (df_clean['horas_mensuales'] > limite_sup)]
print(f"\nValores atípicos detectados en horas_mensuales: {len(outliers)}")
print(f"Rango normal aproximado: {limite_inf:.1f} - {limite_sup:.1f} horas")

# Decisión: los mantenemos porque en streaming es realista
# que existan usuarios con consumo muy alto (binge-watchers)
# Solo los recortamos un poco para que no distorsionen demasiado el escalado
df_clean['horas_mensuales'] = df_clean['horas_mensuales'].clip(upper=limite_sup + 20)

# Verificación de tipos
df_clean['frecuencia_semanal'] = df_clean['frecuencia_semanal'].astype(int)
df_clean['porcentaje_series'] = df_clean['porcentaje_series'].astype(int)

print("\nDatos después de la limpieza:")
print(df_clean.describe().round(2))

# ============================================================
# 3. SELECCIÓN DE VARIABLES Y ESCALAMIENTO
# ============================================================

print("\n" + "=" * 60)
print("3. SELECCIÓN DE VARIABLES Y ESCALAMIENTO")
print("=" * 60)

# Variables elegidas para K-Means:
# 1. horas_mensuales      -> intensidad de uso
# 2. frecuencia_semanal   -> hábito / regularidad
# 3. rating_promedio      -> nivel de satisfacción / exigencia
# 4. porcentaje_series    -> preferencia de formato (series vs películas)

# Justificación: estas cuatro variables capturan el comportamiento
# de consumo sin depender de variables categóricas (que necesitarían
# encoding y complican la interpretación de los centroides).

vars_cluster = ['horas_mensuales', 'frecuencia_semanal',
                'rating_promedio', 'porcentaje_series']

X = df_clean[vars_cluster].copy()
print("Variables seleccionadas:", vars_cluster)
print(X.head())

# Escalamos con StandardScaler (media 0, desviación 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=vars_cluster)

print("\nDatos escalados (primeras filas):")
print(X_scaled.head().round(3))

# ============================================================
# 4. IMPLEMENTACIÓN DE K-MEANS Y SELECCIÓN DE K
# ============================================================

print("\n" + "=" * 60)
print("4. IMPLEMENTACIÓN DE K-MEANS")
print("=" * 60)

# Probamos varios valores de K
rango_k = range(2, 9)
inertias = []
siluetas = []

for k in rango_k:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(X_scaled)
    inertias.append(modelo.inertia_)
    siluetas.append(silhouette_score(X_scaled, etiquetas))
    print(f"K={k} | Inercia={modelo.inertia_:.1f} | Silhouette={siluetas[-1]:.3f}")

# Gráfica del método del codo + silhouette
fig, ax1 = plt.subplots(figsize=(10, 5))

color1 = '#E50914'
ax1.set_xlabel('Número de clusters (K)')
ax1.set_ylabel('Inercia', color=color1)
ax1.plot(rango_k, inertias, 'o-', color=color1, linewidth=2, markersize=8)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = '#221f1f'
ax2.set_ylabel('Silhouette Score', color=color2)
ax2.plot(rango_k, siluetas, 's--', color=color2, linewidth=2, markersize=8)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Método del codo y Silhouette Score')
fig.tight_layout()
plt.savefig('seleccion_k.png', dpi=120)
plt.close()
print("\n[Gráfica guardada] seleccion_k.png")

# Elegimos K=4
# Justificación: el codo se suaviza alrededor de 4 y el silhouette
# se mantiene razonable. Con 4 grupos obtenemos perfiles claros
# y accionables para recomendaciones y retención.
k_optimo = 4
print(f"\n*** K seleccionado: {k_optimo} ***")

# Modelo final
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# Centroides (en escala original para interpretar)
centroides_scaled = kmeans.cluster_centers_
centroides = scaler.inverse_transform(centroides_scaled)
centroides_df = pd.DataFrame(centroides, columns=vars_cluster)
centroides_df.index = [f'Cluster {i}' for i in range(k_optimo)]
print("\nCentroides (valores originales):")
print(centroides_df.round(2))

# Cantidad de usuarios por cluster
print("\nDistribución de usuarios por cluster:")
print(df_clean['cluster'].value_counts().sort_index())

# Gráfica de dispersión (horas vs frecuencia, coloreada por cluster)
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df_clean, x='horas_mensuales', y='frecuencia_semanal',
                hue='cluster', palette='Set1', alpha=0.6, s=40)
plt.title('Agrupamientos K-Means (Horas vs Frecuencia)')
plt.xlabel('Horas mensuales')
plt.ylabel('Frecuencia semanal (días)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('clusters_scatter.png', dpi=120)
plt.close()
print("\n[Gráfica guardada] clusters_scatter.png")

# Comparación rápida de dos configuraciones (K=3 vs K=4)
print("\n--- Comparación K=3 vs K=4 ---")
for k_comp in [3, 4]:
    m = KMeans(n_clusters=k_comp, random_state=42, n_init=10)
    lab = m.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, lab)
    print(f"K={k_comp}: Silhouette={sil:.3f} | Inercia={m.inertia_:.1f}")

# ============================================================
# 5. ANÁLISIS E INTERPRETACIÓN DE RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("5. ANÁLISIS E INTERPRETACIÓN")
print("=" * 60)

# Estadísticas por cluster
print("\nEstadísticas descriptivas por cluster:")
stats = df_clean.groupby('cluster')[vars_cluster].agg(['mean', 'median', 'std']).round(2)
print(stats)

# Perfiles descriptivos (nombres de los clusters)
# Basados en los centroides:
# Cluster 0: consumo medio-bajo, frecuencia baja, rating alto, más películas
# Cluster 1: alto consumo, alta frecuencia, rating medio, mixto
# etc. (ajustamos después de ver los números reales)

print("\n--- Perfiles de los clusters ---")
for i in range(k_optimo):
    subset = df_clean[df_clean['cluster'] == i]
    print(f"\nCluster {i} ({len(subset)} usuarios):")
    print(f"  Horas mensuales promedio : {subset['horas_mensuales'].mean():.1f}")
    print(f"  Frecuencia semanal       : {subset['frecuencia_semanal'].mean():.1f} días")
    print(f"  Rating promedio          : {subset['rating_promedio'].mean():.2f}")
    print(f"  % series                 : {subset['porcentaje_series'].mean():.0f}%")
    print(f"  Suscripción más común    : {subset['tipo_suscripcion'].mode()[0]}")
    print(f"  Género preferido top     : {subset['genero_preferido'].mode()[0]}")

# Tabla comparativa simple
tabla = df_clean.groupby('cluster').agg({
    'horas_mensuales': 'mean',
    'frecuencia_semanal': 'mean',
    'rating_promedio': 'mean',
    'porcentaje_series': 'mean',
    'user_id': 'count'
}).round(2)
tabla.columns = ['Horas_prom', 'Freq_sem', 'Rating', '%_Series', 'N_usuarios']
print("\nTabla comparativa de clusters:")
print(tabla)

# Guardamos el dataset procesado
df_clean.to_csv('netflix_usuarios_con_clusters.csv', index=False, encoding='utf-8')
print("\nDataset procesado guardado: netflix_usuarios_con_clusters.csv")

# Guardamos también los centroides
centroides_df.to_csv('centroides_kmeans.csv')
print("Centroides guardados: centroides_kmeans.csv")

print("\n" + "=" * 60)
print("Proceso completado.")
print("=" * 60)
