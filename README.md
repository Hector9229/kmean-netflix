# Segmentación de usuarios de Netflix con K-Means

Proyecto integrador de minería de datos: identificación de perfiles de consumo de series y películas mediante el algoritmo K-Means y visualización en un dashboard interactivo con Streamlit.

## Descripción del problema

Las plataformas de streaming necesitan entender cómo consumen contenido sus usuarios para mejorar recomendaciones, diseñar estrategias de retención y optimizar la oferta.  
Este proyecto utiliza técnicas de agrupamiento (K-Means) sobre variables de comportamiento (horas de consumo, frecuencia de uso, rating promedio y preferencia series/películas) para identificar segmentos accionables.

**Objetivo general**  
Identificar perfiles de consumo de usuarios de Netflix (series y películas) mediante K-Means y presentar los resultados en un dashboard interactivo que facilite el análisis y la toma de decisiones.

**Objetivos específicos**
1. Explorar y preparar un dataset de usuarios de streaming.
2. Aplicar K-Means evaluando distintos valores de K y seleccionar el más adecuado.
3. Caracterizar e interpretar los clusters obtenidos.
4. Desarrollar un dashboard interactivo (Streamlit) con KPIs, visualizaciones y filtros.

## Dataset

- **Archivo original**: `netflix_usuarios_raw.csv`  
- **Registros**: ~3.000 usuarios sintéticos (realista)  
- **Variables principales**:
  - `user_id`, `edad`, `genero_usuario`, `pais`
  - `tipo_suscripcion` (Basic / Standard / Premium)
  - `meses_activo`
  - `dispositivo_principal` (Smart TV, Móvil, Tablet, PC)
  - `genero_preferido`
  - `horas_mensuales`
  - `frecuencia_semanal`
  - `rating_promedio`
  - `porcentaje_series`

El dataset incluye valores nulos y algunos duplicados para practicar la limpieza.

**Archivo procesado**: `netflix_usuarios_con_clusters.csv` (incluye la columna `cluster`).

## Estructura del repositorio

```
├── netflix_usuarios_raw.csv          # Dataset original
├── netflix_usuarios_con_clusters.csv # Dataset con clusters asignados
├── centroides_kmeans.csv             # Centroides del modelo final
├── analisis_kmeans_netflix.py        # Script completo de análisis
├── app_streamlit.py                  # Dashboard interactivo
├── eda_*.png / seleccion_k.png / clusters_scatter.png  # Gráficas generadas
└── README.md
```

## Requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly
```

## Cómo ejecutar el análisis

```bash
python analisis_kmeans_netflix.py
```

El script realiza:
1. Carga y exploración (EDA)
2. Limpieza (nulos, duplicados, outliers)
3. Escalamiento (StandardScaler)
4. Evaluación de varios K (codo + silhouette)
5. Entrenamiento del modelo final (K=4)
6. Interpretación de clusters y exportación de resultados

## Cómo ejecutar el dashboard

Desde la carpeta del proyecto:

```bash
streamlit run app_streamlit.py
```

El dashboard tiene dos secciones:
- **Análisis general**: KPIs, distribución de horas, suscripciones, géneros y dispositivos.
- **Segmentación K-Means**: distribución por cluster, scatter, boxplots comparativos, tabla de estadísticas, descripción de perfiles, hallazgos y recomendaciones.

Filtros interactivos disponibles: cluster, tipo de suscripción y país.

## Clusters identificados (K = 4)

| Cluster | Nombre descriptivo                      | Características principales                          |
|---------|-----------------------------------------|------------------------------------------------------|
| 0       | Binge Watchers (Alto consumo)           | Muchas horas, alta frecuencia, más Premium           |
| 1       | Series Fans                             | Bajo-medio consumo, alto % de series                 |
| 2       | Cinéfilos exigentes                     | Bajo consumo, rating alto, más películas             |
| 3       | Usuarios moderados                      | Consumo medio, ratings más bajos                     |

## Herramientas utilizadas

- Python 3
- Pandas, NumPy
- Scikit-learn (KMeans, StandardScaler, silhouette_score)
- Matplotlib, Seaborn
- Streamlit + Plotly

## Notas

- El dataset es **sintético** (generado de forma realista) y no contiene datos reales de Netflix.
- El código está comentado en español de forma natural.
- Las gráficas del análisis se guardan automáticamente al ejecutar el script.
