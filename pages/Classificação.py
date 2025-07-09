import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Classificação", layout="wide")
st.title("📈 Classificação")

@st.cache_data
def load_data():
    return pd.read_csv("datasets/spotify_processed.csv", encoding='utf-8', low_memory=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Arquivo 'spotify_processed.csv' não encontrado na pasta 'datasets'")
    st.stop()

st.sidebar.header("🎛 Filtros avançados")

st.sidebar.subheader("Filtros básicos")

popularidade_filter = st.sidebar.selectbox(
    "Filtrar por popularidade:",
    ["Todas", "Populares", "Não populares"]
)

explicit_filter = st.sidebar.selectbox(
    "Conteúdo explícito:",
    ["Todos", "Explícito", "Não explícito"]
)

st.sidebar.subheader("Características musicais")

danceability_range = st.sidebar.slider(
    "Dançabilidade:",
    min_value=float(df['danceability'].min()),
    max_value=float(df['danceability'].max()),
    value=(float(df['danceability'].min()), float(df['danceability'].max())),
    step=0.1
)

energy_range = st.sidebar.slider(
    "Energia:",
    min_value=float(df['energy'].min()),
    max_value=float(df['energy'].max()),
    value=(float(df['energy'].min()), float(df['energy'].max())),
    step=0.1
)

valence_range = st.sidebar.slider(
    "Valência (positividade):",
    min_value=float(df['valence'].min()),
    max_value=float(df['valence'].max()),
    value=(float(df['valence'].min()), float(df['valence'].max())),
    step=0.1
)

acousticness_range = st.sidebar.slider(
    "Acústico:",
    min_value=float(df['acousticness'].min()),
    max_value=float(df['acousticness'].max()),
    value=(float(df['acousticness'].min()), float(df['acousticness'].max())),
    step=0.1
)

duration_range = st.sidebar.slider(
    "Duração (normalizada):",
    min_value=float(df['duration_ms'].min()),
    max_value=float(df['duration_ms'].max()),
    value=(float(df['duration_ms'].min()), float(df['duration_ms'].max())),
    step=0.1
)

st.sidebar.subheader("Características categóricas")

key_columns = [col for col in df.columns if col.startswith('key_')]
selected_keys = st.sidebar.multiselect(
    "Tonalidades:",
    key_columns,
    default=key_columns[:3],
    format_func=lambda x: f"Key {x.split('_')[1]}"
)

df_filtrado = df.copy()

if popularidade_filter == "Populares":
    df_filtrado = df_filtrado[df_filtrado['is_popular'] == 1]
elif popularidade_filter == "Não populares":
    df_filtrado = df_filtrado[df_filtrado['is_popular'] == 0]

if explicit_filter == "Explícito":
    df_filtrado = df_filtrado[df_filtrado['explicit'] > 0]
elif explicit_filter == "Não explícito":
    df_filtrado = df_filtrado[df_filtrado['explicit'] <= 0]

df_filtrado = df_filtrado[
    (df_filtrado['danceability'] >= danceability_range[0]) & 
    (df_filtrado['danceability'] <= danceability_range[1])
]

df_filtrado = df_filtrado[
    (df_filtrado['energy'] >= energy_range[0]) & 
    (df_filtrado['energy'] <= energy_range[1])
]

df_filtrado = df_filtrado[
    (df_filtrado['valence'] >= valence_range[0]) & 
    (df_filtrado['valence'] <= valence_range[1])
]

df_filtrado = df_filtrado[
    (df_filtrado['acousticness'] >= acousticness_range[0]) & 
    (df_filtrado['acousticness'] <= acousticness_range[1])
]

df_filtrado = df_filtrado[
    (df_filtrado['duration_ms'] >= duration_range[0]) & 
    (df_filtrado['duration_ms'] <= duration_range[1])
]

if selected_keys:
    key_filter = df_filtrado[selected_keys].sum(axis=1) > 0
    df_filtrado = df_filtrado[key_filter]

total_musicas = len(df_filtrado)
pct_total = total_musicas/len(df)*100
st.info(f"📊 **{total_musicas:,}** músicas selecionadas de **{len(df):,}** total ({pct_total:.1f}%)")

if len(df_filtrado) == 0:
    st.warning("⚠️ Nenhuma música encontrada com os filtros aplicados. Tente ajustar os critérios.")
    st.stop()

st.header("🔍 Análise comparativa para Classificação")

tab1, tab2 = st.tabs(["🎯 Correlações", "📊 Comparações"])

with tab1:
    st.subheader("Análise de correlações")
    
    caracteristicas_disponiveis = ['danceability', 'energy', 'valence', 'acousticness', 
                                 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo']

    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox(
            "Eixo X:",
            caracteristicas_disponiveis,
            index=0,
            key="class_x_axis"
        )
    
    with col2:
        y_axis = st.selectbox(
            "Eixo Y:",
            caracteristicas_disponiveis,
            index=1,
            key="class_y_axis"
        )
    
    color_by_popularity = st.checkbox("Colorir por popularidade", value=True, key="class_color_pop")
    
    sample_size = min(2000, len(df_filtrado))
    df_sample = df_filtrado.sample(sample_size) if len(df_filtrado) > sample_size else df_filtrado
    
    if color_by_popularity:
        fig_scatter = px.scatter(
            df_sample,
            x=x_axis,
            y=y_axis,
            color='is_popular',
            title=f"Correlação: {x_axis.capitalize()} vs {y_axis.capitalize()}",
            labels={'is_popular': 'Popular'},
            color_discrete_map={0: '#FF6B6B', 1: '#1DB954'}
        )
    else:
        fig_scatter = px.scatter(
            df_sample,
            x=x_axis,
            y=y_axis,
            title=f"Correlação: {x_axis.capitalize()} vs {y_axis.capitalize()}"
        )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    corr_coef = df_filtrado[x_axis].corr(df_filtrado[y_axis])
    st.metric("Coeficiente de correlação", f"{corr_coef:.3f}")

with tab2:
    st.subheader("Comparação entre categorias")
    
    char_comparacao = st.selectbox(
        "Característica para comparação:",
        caracteristicas_disponiveis,
        key="class_comp_char"
    )
    
    categoria_comparacao = st.selectbox(
        "Comparar por:",
        ['is_popular', 'explicit'],
        format_func=lambda x: 'Popularidade' if x == 'is_popular' else 'Conteúdo explícito',
        key="class_cat_comp"
    )
    
    fig_box = px.box(
        df_filtrado,
        x=categoria_comparacao,
        y=char_comparacao,
        title=f"Comparação de {char_comparacao.capitalize()} por {('popularidade' if categoria_comparacao == 'is_popular' else 'conteúdo')}",
        labels={categoria_comparacao: 'Categoria', char_comparacao: char_comparacao.capitalize()}
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.subheader("Estatísticas descritivas")
    stats_df = df_filtrado.groupby(categoria_comparacao)[char_comparacao].describe().round(3)
    st.dataframe(stats_df, use_container_width=True)

st.header("💡 Resumo dos dados filtrados para Classificação")

col1, col2, col3 = st.columns(3)

with col1:
    populares = df_filtrado['is_popular'].sum()
    pct_populares_filtro = (populares / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
    st.metric(
        "🔥 Músicas populares",
        f"{populares:,}",
        f"{pct_populares_filtro:.1f}% das músicas filtradas"
    )

with col2:
    explicitas = (df_filtrado['explicit'] > 0).sum()
    pct_explicitas_filtro = (explicitas / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
    st.metric(
        "🚫 Conteúdo explícito",
        f"{explicitas:,}",
        f"{pct_explicitas_filtro:.1f}% das músicas filtradas"
    )

with col3:
    st.metric(
        "💃 Dançabilidade média",
        f"{df_filtrado['danceability'].mean():.3f}",
        f"±{df_filtrado['danceability'].std():.3f}"
    )

st.markdown("---")
st.markdown("*💡 Use os filtros na barra lateral para explorar diferentes perfis musicais*")
