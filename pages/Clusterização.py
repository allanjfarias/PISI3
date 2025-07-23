import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Clusterização", layout="wide")
st.title("🧩 Clusterização")

@st.cache_data
def load_data():
    return pd.read_csv("datasets_paralelos/spotify_limpo.csv", encoding='utf-8', low_memory=True)
try:
    df = load_data()
    if 'is_popular' not in df.columns:
        df['is_popular'] = df['popularity'].apply(lambda x: 1 if x >= 70 else 0)
except FileNotFoundError:
    st.error("❌ Arquivo 'spotify_limpo.csv' não encontrado na pasta 'datasets_paralelos'")
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
    "Positividade:",
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

total_musicas = len(df_filtrado)
pct_total = total_musicas/len(df)*100
st.info(f"📊 **{total_musicas:,}** músicas selecionadas de **{len(df):,}** total ({pct_total:.1f}%)")

if len(df_filtrado) == 0:
    st.warning("⚠️ Nenhuma música encontrada com os filtros aplicados. Tente ajustar os critérios.")
    st.stop()

st.header("📊 Gráficos")

caracteristicas_disponiveis = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo']

col1, col2 = st.columns(2)

with col1:
    eixo_x = st.selectbox("Variável no eixo X:", caracteristicas_disponiveis, index=0)

with col2:
    eixo_y = st.selectbox("Variável no eixo Y:", caracteristicas_disponiveis, index=1)

n_clusters = st.slider("Número de clusters", min_value=1, max_value=10, value=3)

if len(df_filtrado) >= n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_filtrado['cluster'] = kmeans.fit_predict(df_filtrado[[eixo_x, eixo_y]])

    fig_cluster = px.scatter(
        df_filtrado,
        x=eixo_x,
        y=eixo_y,
        color='cluster',
        title=f"Clusterização: {eixo_x} vs {eixo_y}",
        hover_data=['track_name', 'artists']
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
else:
    st.warning("⚠️ Número de clusters maior que número de músicas disponíveis.")

st.header("💡 Resumo dos dados filtrados para Clusterização")

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

st.header("🎯 Perfil musical dos dados filtrados")

if len(df_filtrado) > 0:
    caracteristicas_media = df_filtrado[['danceability', 'energy', 'valence', 'acousticness']].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("💃 Dançabilidade", f"{caracteristicas_media['danceability']:.3f}")
    with col2:
        st.metric("⚡ Energia", f"{caracteristicas_media['energy']:.3f}")
    with col3:
        st.metric("😊 Positividade", f"{caracteristicas_media['valence']:.3f}")
    with col4:
        st.metric("🎸 Acústico", f"{caracteristicas_media['acousticness']:.3f}")

if st.button("📥 Exportar dados filtrados"):
    csv = df_filtrado.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"spotify_filtrado_{len(df_filtrado)}_musicas.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("*💡 Use os filtros na barra lateral para explorar diferentes perfis musicais*")
