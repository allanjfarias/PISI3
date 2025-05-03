import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Análise Spotify 900k", layout="wide")
st.title("Análise Exploratória - Dataset Spotify 900k")


@st.cache_data
def load_data():
    df = pd.read_csv("datasets/spotify.csv")
    return df


df = load_data()

st.subheader("1. Informações Estruturais")
st.write("Formato (linhas, colunas):", df.shape)
st.write("Colunas e Tipos de Dados:")
info_df = pd.DataFrame(
    {"Coluna": df.columns, "Tipo de Dado": df.dtypes.values})
st.dataframe(info_df)

with st.expander("Visualizar primeiras linhas do dataset"):
    st.dataframe(df.head())

st.subheader("2. Qualidade dos Dados")


nulos = df.isnull().sum().reset_index()
nulos.columns = ["Coluna", "Qtd. Nulos"]
st.write('Nulos por coluna')
st.dataframe(nulos)


st.subheader("3. Resumo Estatístico")
with st.expander("describe()"):
    st.dataframe(df.describe())


st.subheader("4. Exploração Inicial")
with st.expander("Quantidade de músicas explícitas"):
    st.write("Distribuição de músicas explícitas:")
    st.plotly_chart(px.histogram(df, x="Explicit", color="Explicit"))


with st.expander("Distribuição da Popularidade"):
    st.write("Distribuição da Popularidade:")
    st.plotly_chart(px.histogram(df, x="Popularity", nbins=50))
#
# Gráficos de Márcia
#

# Top artistas com mais músicas
with st.expander("Artistas com mais músicas"):
    st.write("Top 5 artistas com mais músicas:")
    top_artistas = df['Artist(s)'].value_counts().nlargest(5).reset_index()
    top_artistas.columns = ['Artista', 'Qtd. de Músicas']
    st.plotly_chart(px.bar(top_artistas, x="Artista",
                    y="Qtd. de Músicas", color="Artista"))

# Energy x Danceability
with st.expander("Energy x Danceability"):
    st.write("Correlação entre Energy e Danceability:")
    st.plotly_chart(px.scatter(df, x="Energy", y="Danceability",
                    color="Popularity", title="Energy x Danceability"))
