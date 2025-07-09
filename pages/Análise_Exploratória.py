import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Análise Exploratória", layout="wide")
st.title("🎵 Análise Exploratória de Dados (EDA)")

@st.cache_data
def load_data():
    return pd.read_csv("datasets/spotify_processed.csv", encoding='utf-8', low_memory=True)

try:
    df = load_data()
    st.success(f"✅ Dataset carregado com sucesso: {df.shape[0]:,} músicas e {df.shape[1]} características")
except FileNotFoundError:
    st.error("❌ Arquivo 'spotify_processed.csv' não encontrado na pasta 'datasets'")
    st.stop()

st.sidebar.header("🎛️ Filtros globais")

popularidade_filter = st.sidebar.selectbox(
    "Filtrar por popularidade:",
    ["Todas", "Populares", "Não populares"]
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

explicit_filter = st.sidebar.selectbox(
    "Conteúdo explícito:",
    ["Todos", "Explícito", "Não explícito"],
    key="explicit_filter_eda"
)
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

st.info(f"📊 Dados filtrados: {df_filtrado.shape[0]:,} músicas selecionadas")

st.header("📊 1. Visão geral dos dados")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎵 Total de músicas", f"{df_filtrado.shape[0]:,}")
with col2:
    populares = df_filtrado['is_popular'].sum()
    st.metric("🔥 Músicas populares", f"{populares:,}")
with col3:
    pct_populares = (populares / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
    st.metric("📈 Percentual popular", f"{pct_populares:.1f}%")
with col4:
    st.metric("🎼 Características", df_filtrado.shape[1])

st.header("2. Análise de popularidade")

pop_counts = df_filtrado['is_popular'].value_counts()
fig_pop = px.pie(
    values=pop_counts.values,
    names=['Não popular', 'Popular'],
    title="Distribuição de popularidade",
    color_discrete_sequence=['#FF6B6B', '#1DB954']
)
st.plotly_chart(fig_pop, use_container_width=True)

st.subheader("Características médias por popularidade")
caracteristicas_principais = ['danceability', 'energy', 'valence', 'acousticness']

dados_comparacao = []
for char in caracteristicas_principais:
    for pop in [0, 1]:
        dados_comparacao.append({
            'Característica': char.capitalize(),
            'Popularidade': 'Popular' if pop == 1 else 'Não popular',
            'Valor': df_filtrado[df_filtrado['is_popular'] == pop][char].mean()
        })

df_comp = pd.DataFrame(dados_comparacao)

fig_comp = px.bar(
    df_comp,
    x='Característica',
    y='Valor',
    color='Popularidade',
    title="Características médias por popularidade",
    barmode='group',
    color_discrete_sequence=['#FF6B6B', '#1DB954']
)
st.plotly_chart(fig_comp, use_container_width=True)

st.header("🎼 3. Características musicais")

caracteristicas_musicais = [
    'danceability', 'energy', 'valence', 'acousticness', 
    'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo'
]

caracteristica_selecionada = st.selectbox(
    "Selecione uma característica musical para análise detalhada:",
    caracteristicas_musicais,
    format_func=lambda x: x.capitalize()
)

fig_char = px.histogram(
    df_filtrado,
    x=caracteristica_selecionada,
    nbins=25,
    title=f"Distribuição de {caracteristica_selecionada.capitalize()}",
    labels={caracteristica_selecionada: caracteristica_selecionada.capitalize()},
    color_discrete_sequence=['#FF6B6B']
)
st.plotly_chart(fig_char, use_container_width=True)

fig_box = px.box(
    df_filtrado,
    x='is_popular',
    y=caracteristica_selecionada,
    title=f"{caracteristica_selecionada.capitalize()} por popularidade",
    labels={'is_popular': 'Popularidade (0=Não, 1=Sim)'}, 
    color_discrete_map={0: '#FF6B6B', 1: '#1DB954'}
)
st.plotly_chart(fig_box, use_container_width=True)

st.header("🔍 4. Correlações")

caracteristicas_numericas = ['danceability', 'energy', 'valence', 'acousticness', 
                           'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo', 'duration_ms']

caracteristicas_existentes = [col for col in caracteristicas_numericas if col in df_filtrado.columns]

corr_matrix = df_filtrado[caracteristicas_existentes].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title="Matriz de correlação - características musicais",
    color_continuous_scale='RdBu_r'
)
fig_corr.update_layout(height=600)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("📊 Estatísticas por popularidade")

if len(df_filtrado) > 0:
    stats_populares = df_filtrado[df_filtrado['is_popular'] == 1][caracteristicas_existentes].mean()
    stats_nao_populares = df_filtrado[df_filtrado['is_popular'] == 0][caracteristicas_existentes].mean()
    
    stats_df = pd.DataFrame({
        'Não populares': stats_nao_populares,
        'Populares': stats_populares,
        'Diferença': stats_populares - stats_nao_populares
    }).round(3)
    
    st.dataframe(stats_df, use_container_width=True)

st.markdown("---")
st.markdown("*💡 Use os filtros na barra lateral para explorar diferentes segmentos dos dados*")
