import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib as plt
# Configuração inicial da página
st.set_page_config(page_title="Análise Spotify 900k", layout="wide")
st.title("🎵 Análise Exploratória - Dataset Spotify 900k")



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
