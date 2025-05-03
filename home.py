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
 
    
with st.expander("Popularidade vs Positiveness"):
    fig = px.scatter(df, x="Positiveness", y="Popularity", color="Explicit",
                     title="Popularidade vs Positiveness", opacity=0.6,
                     labels={"Positiveness": "Positividade", "Popularity": "Popularidade"})
    st.plotly_chart(fig)


with st.expander("Distribuição de Duração das Músicas"):
    st.subheader("Distribuição do Comprimento das Músicas (Length)")
    df_sorted = df.sort_values(by="Length")  # Ordenar o DataFrame pela coluna "Length"
    fig = px.histogram(df_sorted, x="Length", nbins=50, title="Distribuição do Comprimento das Músicas",
                       labels={"Length": "Duração (segundos)"}, color_discrete_sequence=["#636EFA"])
    fig.update_layout(bargap=0.1, xaxis_title="Duração (segundos)", yaxis_title="Quantidade")
    st.plotly_chart(fig)

with st.expander("Top 10 Artistas Mais Populares (Média de Popularidade)"):
    top_artistas_pop = df.groupby('Artist(s)')['Popularity'].mean(
    ).sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_artistas_pop, x='Artist(s)', y='Popularity', color='Popularity',
                 title="Top 10 Artistas com Maior Popularidade Média",
                 labels={'Artist(s)': 'Artista', 'Popularity': 'Popularidade Média'})
    st.plotly_chart(fig)

with st.expander("Danceability por Gênero (Boxplot)"):
    df_genres = df[df['Genre'].isin(
        df['Genre'].value_counts().head(8).index)]  # Top 8 gêneros
    fig = px.box(df_genres, x='Genre', y='Danceability', color='Genre',
                 title="Distribuição de Danceability por Gênero",
                 labels={'Genre': 'Gênero', 'Danceability': 'Danceabilidade'})
    st.plotly_chart(fig)