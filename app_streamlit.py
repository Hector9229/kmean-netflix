# -*- coding: utf-8 -*-
"""
Dashboard interactivo - Segmentación de usuarios Netflix
Proyecto Integrador: K-Means + Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Segmentación Netflix | K-Means",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E50914;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #564d4d;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #E50914;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #221f1f;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #564d4d;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv('netflix_usuarios_con_clusters.csv')
    return df

df = cargar_datos()

# Nombres descriptivos de los clusters (basados en el análisis)
nombres_cluster = {
    0: "Binge Watchers (Alto consumo)",
    1: "Series Fans (Bajo consumo, alto % series)",
    2: "Cinéfilos exigentes (Bajo consumo, alto rating)",
    3: "Usuarios moderados (Rating bajo-medio)"
}

df['nombre_cluster'] = df['cluster'].map(nombres_cluster)

# ----------------------------------------------------------
# Sidebar - Filtros
# ----------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=160)
st.sidebar.title("Filtros")

# Filtro 1: Cluster
clusters_sel = st.sidebar.multiselect(
    "Cluster",
    options=sorted(df['cluster'].unique()),
    default=sorted(df['cluster'].unique()),
    format_func=lambda x: f"{x} - {nombres_cluster[x]}"
)

# Filtro 2: Tipo de suscripción
subs_sel = st.sidebar.multiselect(
    "Tipo de suscripción",
    options=df['tipo_suscripcion'].unique().tolist(),
    default=df['tipo_suscripcion'].unique().tolist()
)

# Filtro 3: País
paises_sel = st.sidebar.multiselect(
    "País",
    options=sorted(df['pais'].unique()),
    default=sorted(df['pais'].unique())
)

# Aplicar filtros
df_filtrado = df[
    (df['cluster'].isin(clusters_sel)) &
    (df['tipo_suscripcion'].isin(subs_sel)) &
    (df['pais'].isin(paises_sel))
]

# ----------------------------------------------------------
# Navegación por páginas
# ----------------------------------------------------------
pagina = st.sidebar.radio(
    "Navegación",
    ["📊 Análisis general", "🎯 Segmentación K-Means"],
    index=0
)

# ============================================================
# PÁGINA 1 - ANÁLISIS GENERAL
# ============================================================
if pagina == "📊 Análisis general":

    st.markdown('<p class="main-header">Segmentación de usuarios de Netflix</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis de patrones de consumo de series y películas</p>', unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{len(df_filtrado):,}</div>
            <div class="kpi-label">Usuarios analizados</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        horas_prom = df_filtrado['horas_mensuales'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{horas_prom:.1f} h</div>
            <div class="kpi-label">Horas mensuales promedio</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        rating_prom = df_filtrado['rating_promedio'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{rating_prom:.2f}</div>
            <div class="kpi-label">Rating promedio</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        pct_premium = (df_filtrado['tipo_suscripcion'] == 'Premium').mean() * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{pct_premium:.1f}%</div>
            <div class="kpi-label">Usuarios Premium</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráficas
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribución de horas de consumo")
        fig_hist = px.histogram(
            df_filtrado, x='horas_mensuales', nbins=30,
            color_discrete_sequence=['#E50914'],
            labels={'horas_mensuales': 'Horas mensuales', 'count': 'Usuarios'}
        )
        fig_hist.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.subheader("Tipo de suscripción")
        cont_sub = df_filtrado['tipo_suscripcion'].value_counts().reset_index()
        cont_sub.columns = ['tipo', 'cantidad']
        fig_sub = px.pie(
            cont_sub, values='cantidad', names='tipo',
            color_discrete_sequence=['#E50914', '#b20710', '#221f1f'],
            hole=0.4
        )
        fig_sub.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(fig_sub, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Géneros preferidos")
        cont_gen = df_filtrado['genero_preferido'].value_counts().reset_index()
        cont_gen.columns = ['genero', 'cantidad']
        fig_gen = px.bar(
            cont_gen, x='cantidad', y='genero', orientation='h',
            color_discrete_sequence=['#E50914'],
            labels={'cantidad': 'Usuarios', 'genero': ''}
        )
        fig_gen.update_layout(margin=dict(t=20, b=20), height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_gen, use_container_width=True)

    with c4:
        st.subheader("Dispositivo principal")
        cont_disp = df_filtrado['dispositivo_principal'].value_counts().reset_index()
        cont_disp.columns = ['dispositivo', 'cantidad']
        fig_disp = px.bar(
            cont_disp, x='dispositivo', y='cantidad',
            color_discrete_sequence=['#b20710'],
            labels={'cantidad': 'Usuarios', 'dispositivo': ''}
        )
        fig_disp.update_layout(margin=dict(t=20, b=20), height=400)
        st.plotly_chart(fig_disp, use_container_width=True)

# ============================================================
# PÁGINA 2 - SEGMENTACIÓN K-MEANS
# ============================================================
else:

    st.markdown('<p class="main-header">Resultados de la segmentación K-Means</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Perfiles de consumo identificados (K = 4)</p>', unsafe_allow_html=True)

    # KPIs de segmentación
    col1, col2, col3, col4 = st.columns(4)
    for i, col in enumerate([col1, col2, col3, col4]):
        n = len(df_filtrado[df_filtrado['cluster'] == i])
        pct = n / len(df_filtrado) * 100 if len(df_filtrado) > 0 else 0
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{n}</div>
                <div class="kpi-label">Cluster {i}<br><small>{pct:.1f}% del total</small></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Distribución por cluster
    st.subheader("Distribución de usuarios por cluster")
    cont_cl = df_filtrado['cluster'].value_counts().sort_index().reset_index()
    cont_cl.columns = ['cluster', 'cantidad']
    cont_cl['nombre'] = cont_cl['cluster'].map(nombres_cluster)

    fig_dist = px.bar(
        cont_cl, x='cluster', y='cantidad', text='cantidad',
        color='cluster', color_continuous_scale='Reds',
        labels={'cluster': 'Cluster', 'cantidad': 'Usuarios'},
        hover_data=['nombre']
    )
    fig_dist.update_layout(showlegend=False, height=350, margin=dict(t=20))
    fig_dist.update_traces(textposition='outside')
    st.plotly_chart(fig_dist, use_container_width=True)

    # Scatter de clusters
    st.subheader("Visualización de agrupamientos")
    fig_scatter = px.scatter(
        df_filtrado, x='horas_mensuales', y='frecuencia_semanal',
        color='nombre_cluster',
        size='rating_promedio',
        hover_data=['genero_preferido', 'tipo_suscripcion', 'pais'],
        labels={
            'horas_mensuales': 'Horas mensuales',
            'frecuencia_semanal': 'Frecuencia semanal (días)',
            'nombre_cluster': 'Cluster'
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.65
    )
    fig_scatter.update_layout(height=450, margin=dict(t=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Comparación de características
    st.subheader("Comparación de características entre clusters")

    c1, c2 = st.columns(2)

    with c1:
        # Boxplot horas
        fig_box = px.box(
            df_filtrado, x='cluster', y='horas_mensuales',
            color='cluster', color_discrete_sequence=px.colors.qualitative.Set1,
            labels={'horas_mensuales': 'Horas mensuales', 'cluster': 'Cluster'}
        )
        fig_box.update_layout(showlegend=False, height=350, title='Horas mensuales por cluster')
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        # Boxplot rating
        fig_box2 = px.box(
            df_filtrado, x='cluster', y='rating_promedio',
            color='cluster', color_discrete_sequence=px.colors.qualitative.Set1,
            labels={'rating_promedio': 'Rating promedio', 'cluster': 'Cluster'}
        )
        fig_box2.update_layout(showlegend=False, height=350, title='Rating promedio por cluster')
        st.plotly_chart(fig_box2, use_container_width=True)

    # Tabla de estadísticas
    st.subheader("Estadísticas descriptivas por cluster")
    stats = df_filtrado.groupby(['cluster', 'nombre_cluster']).agg({
        'horas_mensuales': ['mean', 'median'],
        'frecuencia_semanal': 'mean',
        'rating_promedio': 'mean',
        'porcentaje_series': 'mean',
        'user_id': 'count'
    }).round(2)
    stats.columns = ['Horas (media)', 'Horas (mediana)', 'Freq. semanal', 'Rating', '% Series', 'N usuarios']
    stats = stats.reset_index()
    st.dataframe(stats, use_container_width=True, hide_index=True)

    # Descripción de grupos
    st.markdown("---")
    st.subheader("Descripción de los grupos identificados")

    descripciones = {
        0: """
        **Binge Watchers (Alto consumo)**  
        Usuarios con el mayor volumen de horas mensuales y frecuencia de uso elevada. 
        Predominan las suscripciones Premium. Son el segmento más valioso para retención 
        y upselling de contenido original.
        """,
        1: """
        **Series Fans (Bajo-medio consumo, alto % series)**  
        Prefieren series sobre películas. Consumo moderado y ratings relativamente altos. 
        Ideal para campañas de recomendaciones de temporadas y lanzamientos de series.
        """,
        2: """
        **Cinéfilos exigentes (Bajo consumo, alto rating)**  
        Ven menos contenido pero lo califican mejor. Mayor proporción de películas. 
        Responden bien a catálogo de calidad y recomendaciones personalizadas de cine.
        """,
        3: """
        **Usuarios moderados (Rating bajo-medio)**  
        Consumo similar al promedio pero con ratings más bajos. Posible riesgo de churn. 
        Requieren estrategias de engagement y mejora de la experiencia de recomendación.
        """
    }

    for i in sorted(df_filtrado['cluster'].unique()):
        with st.expander(f"Cluster {i}: {nombres_cluster[i]}"):
            st.markdown(descripciones[i])

    # Hallazgos y recomendaciones
    st.markdown("---")
    st.subheader("Principales hallazgos y recomendaciones")

    st.markdown("""
    **Hallazgos relevantes**
    1. Existe un grupo claro de usuarios de alto consumo (Cluster 0) que concentra 
       una gran parte del tiempo de visionado y tiende a tener planes Premium.
    2. Los usuarios se diferencian notablemente por su preferencia de formato 
       (series vs películas) y por el nivel de satisfacción (rating).
    3. Un segmento importante muestra ratings bajos a pesar de un consumo moderado, 
       lo que puede anticipar problemas de retención.

    **Recomendaciones**
    1. **Cluster 0**: Ofrecer contenido exclusivo y notificaciones de estrenos 
       para mantener el alto engagement.
    2. **Cluster 1 y 2**: Personalizar el home según preferencia de series o películas 
       y destacar contenido bien valorado.
    3. **Cluster 3**: Implementar acciones de reactivación (listas personalizadas, 
       recordatorios, prueba de perfiles) para elevar la satisfacción y reducir churn.

    **Limitaciones del modelo**
    - Se utilizaron solo variables numéricas de comportamiento; variables categóricas 
      (género, dispositivo, país) se usaron de forma descriptiva.
    - El dataset es sintético, por lo que los patrones reflejan supuestos realistas 
      pero no datos reales de Netflix.
    - K-Means asume clusters esféricos y de tamaño similar; perfiles más complejos 
      podrían requerir otros algoritmos.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Proyecto Integrador · K-Means · Streamlit")
st.sidebar.caption("Dataset sintético de usuarios de streaming")
