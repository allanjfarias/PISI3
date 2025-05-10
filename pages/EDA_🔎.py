import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib as plt
# Configuração inicial da página
st.set_page_config(page_title="Análise Spotify Tracks", layout="wide")
st.title("🎵 Análise Exploratória - Spotify Tracks Dataset")



@st.cache_data
def load_data():
    return pd.read_csv("datasets/dataset.csv", encoding='utf-8', low_memory=True)


df = load_data()


st.subheader("🎯 1. Informações Estruturais")
st.info(f"📊 Formato do Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Track Names", df['track_name'].nunique())
col2.metric("Genres", df['track_genre'].nunique())
col3.metric("Artists", df['artists'].nunique())
col4.metric("Albums", df['album_name'].nunique())


st.write("**Colunas e Tipos de Dados:**")
info_df = pd.DataFrame({
    "Coluna": df.columns,
    "Tipo de Dado": df.dtypes.values
})
st.dataframe(info_df)


with st.expander(" Visualizar primeiras linhas do dataset"):
    st.dataframe(df.head())

st.subheader("🧼 2. Qualidade dos Dados")

nulos = df.isnull().sum().reset_index()
nulos.columns = ["Coluna", "Qtd. Nulos"]
st.write("**Quantidade de valores nulos por coluna:**")
st.dataframe(nulos)


st.subheader("📊 3. Resumo Estatístico")

with st.expander(" Resumo das variáveis numéricas (describe):"):
    st.dataframe(df.describe())


st.subheader("🔍 4. Exploração Inicial")

with st.expander('Histogramas para atributos musicais'):
    fig = px.histogram(df, x='popularity', nbins=50, title="Distribuição de Popularidade das Músicas",
                       labels={'popularity': 'Popularidade'},
                       color_discrete_sequence=['skyblue'])
    st.plotly_chart(fig)

    st.markdown("""
    Este gráfico interativo mostra a distribuição das principais características acústicas das músicas do dataset. 
    Você pode visualizar como as músicas variam em relação a cada uma das características, como dançabilidade, energia, valência, entre outras.
    """)

    acoustic_features = [
        'danceability', 'energy', 'valence', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'tempo'
    ]

    for feature in acoustic_features:
        fig = px.histogram(df, x=feature, nbins=50, title=f'Distribuição de {feature.capitalize()}',
                           labels={feature: feature.capitalize()},
                           color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig)

with st.expander("Gêneros Mais Populares"):
    st.write("Este gráfico mostra os gêneros musicais mais populares com base na média de popularidade.")

    genero_popularidade = df.groupby('track_genre')['popularity'].mean().reset_index()

    genero_popularidade = genero_popularidade.sort_values(by='popularity', ascending=False)

    fig = px.bar(genero_popularidade.head(10),  
                 x='track_genre', 
                 y='popularity', 
                 title="Top 10 Gêneros Mais Populares",
                 labels={'track_genre': 'Gênero Musical', 'popularity': 'Popularidade Média'},
                 color='popularity',
                 color_continuous_scale='viridis')

    st.plotly_chart(fig)

with st.expander("Gênero com Mais Músicas no Top 0,001% Mais Populares"):
    st.write("Este gráfico mostra qual gênero tem o maior número de músicas no top 0,001% das músicas mais populares.")

    limite_top = df['popularity'].quantile(0.999)

    top_musicas = df[df['popularity'] >= limite_top]

    genero_top = top_musicas['track_genre'].value_counts().reset_index()
    genero_top.columns = ['track_genre', 'count']

    fig = px.bar(genero_top, 
                 x='count', 
                 y='track_genre', 
                 title="Gêneros com Mais Músicas no Top 0,001% Mais Populares",
                 labels={'track_genre': 'Gênero Musical', 'count': 'Número de Músicas'},
                 color='count',
                 color_continuous_scale='plasma')

    st.plotly_chart(fig)

with st.expander("Músicas Mais Populares"):
    st.write("Este gráfico mostra as músicas mais populares, com base na pontuação de popularidade.")

    top_tracks = df.sort_values(by='popularity', ascending=False).drop_duplicates(subset='track_name').head(20)
    top_tracks['musica_artista'] = top_tracks['track_name'] + ' - ' + top_tracks['artists']

    fig = px.bar(top_tracks.sort_values(by='popularity'),
                 x='popularity',
                 y='musica_artista',
                 orientation='h',
                 title="Top 20 Músicas Mais Populares",
                 labels={'musica_artista': 'Música - Artista', 'popularity': 'Popularidade'},
                 color='popularity',
                 color_continuous_scale='sunset',
                 height=600)

    st.plotly_chart(fig)
