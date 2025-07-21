import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Classificação e Clusterização Aplicadas à Personalização Musical",
    layout="wide",
    menu_items={
        'About': '''Trabalho de Machine Learning e Data Science para a disciplina Projeto Interdisciplinar em Sistemas de Informação III, do curso de Bacharelado em Sistemas de Informação da UFRPE.'''
    }
)

st.title("Classificação e Clusterização Aplicadas à Personalização Musical: Um Estudo de Caso com o TuneTrail")
st.subheader("Análise de dados e modelos de aprendizado de máquina para explorar a personalização musical")

st.markdown("--- ")

st.header("🔍 Perguntas Norteadoras da Pesquisa")
st.markdown("**Pergunta 1: Quais atributos musicais, presentes no dataset \"Spotify Tracks\", são mais relevantes para a formação de clusters de músicas com características semelhantes?**")
st.write("O objetivo desta pergunta é identificar os atributos musicais (danceability, energy, tempo, etc.) que mais contribuem para a formação de grupos de músicas com características similares. A resposta a esta pergunta permitirá a criação de um sistema de recomendação musical que leve em consideração os atributos mais relevantes para a similaridade musical, oferecendo sugestões mais precisas e personalizadas, como demonstrado no estudo de caso do Tune Trail.")
st.markdown("**Pergunta 2: É possível prever se uma música será considerada popular com base em suas características acústicas, como dançabilidade, energia, valência e outras variáveis numéricas disponíveis no conjunto de dados?**")
st.write("O objetivo desta pergunta é investigar a relação entre as características acústicas de uma música e sua popularidade, utilizando métodos de aprendizado de máquina para prever essa variável. Essa análise pode ser aplicada na construção de sistemas de recomendação e playlists em plataformas de streaming, como demonstrado em projetos anteriores. Estudos anteriores indicam que variáveis como dançabilidade, energia e positividade têm correlação significativa com a popularidade das músicas, o que sugere que modelos preditivos baseados nessas características podem ser eficazes na tarefa de previsão.")

st.markdown("--- ")

st.header("📂 Explore as páginas do projeto")
st.write("Navegue pelas seções abaixo para acompanhar a análise de dados e os resultados obtidos.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link("pages/Análise_Exploratória.py", label="📊 Análise exploratória", use_container_width=True)
with col2:
    st.page_link("pages/Classificação.py", label="📈 Classificação", use_container_width=True)
with col3:
    st.page_link("pages/Clusterização.py", label="🧩 Clusterização", use_container_width=True)
with col4:
    st.page_link("pages/Predição.py", label="🤖 Predição", use_container_width=True)

st.markdown("--- ")

st.header("Introdução ao contexto")
st.write("""
Este estudo tem como objetivo apresentar uma análise sobre a aplicação de técnicas de classificação e clusterização na personalização da experiência musical, utilizando o sistema Tune Trail. O dataset utilizado contém informações sobre diversos atributos musicais os quais contribuem para a caracterização e agrupamento de músicas.
""")

st.markdown("--- ")

st.header("Pré-processamento dos Dados")

@st.cache_data
def get_dataset_info():
    df = pd.read_csv("datasets_paralelos/spotify_limpo.csv")
    total_rows = df.shape[0]
    train_size = int(total_rows * 0.8)
    test_size = total_rows - train_size
    return train_size, test_size

train_rows, test_rows = get_dataset_info()

if train_rows is not None and test_rows is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Tamanho do Conjunto de Treino", value=f"{train_rows:,} linhas")
    with col2:
        st.metric(label="Tamanho do Conjunto de Teste", value=f"{test_rows:,} linhas")
else:
    st.error("Não foi possível carregar os dados para determinar o tamanho dos conjuntos de treino e teste.")

st.markdown("--- ")

st.header("Resumo dos Clusters")

cluster_profiles = pd.DataFrame({
    'cluster': [0, 1, 2, 3, 4, 5, 6],
    'danceability': [-0.505468, -0.181864, -0.276515, -1.177324, 0.136966, 0.075905, 0.831382],
    'energy': [0.727698, -0.996020, 0.476356, -1.786997, 0.421410, 0.162316, 0.307871],
    'loudness': [0.657918, -0.460975, 0.283224, -2.311938, -0.002787, -0.540275, 0.393332],
    'speechiness': [-0.015168, -0.291946, 0.000290, -0.317372, -0.150652, 6.511476, 0.074127],
    'acousticness': [-0.744072, 1.019973, -0.098142, 1.558480, -0.667421, 1.228519, -0.301056],
    'valence': [-0.348809, -0.254699, 0.169736, -1.081811, -0.454004, -0.097340, 0.871672],
    'tempo': [0.612328, -0.295871, 0.025672, -0.627150, 0.155983, -0.715480, -0.133923],
    'liveness': [-0.109077, -0.296609, 2.658561, -0.288383, -0.253184, 2.285250, -0.275577],
    'instrumentalness': [-0.436033, -0.450690, -0.311696, 1.787800, 1.831179, -0.535265, -0.490007]
})

popularity_distribution = pd.DataFrame({
    'cluster': [0, 1, 2, 3, 4, 5, 6],
    '0': [0.877980, 0.887610, 0.905074, 0.880977, 0.867098, 0.895844, 0.920063],
    '1': [0.122020, 0.112390, 0.094926, 0.119023, 0.132902, 0.104156, 0.079937]
})

st.subheader("Perfil médio dos clusters (em features processadas)")
st.dataframe(cluster_profiles.set_index('cluster'), use_container_width=True)

st.subheader("Distribuição de Popularidade (proporção) por Cluster")
st.dataframe(popularity_distribution.set_index('cluster'), use_container_width=True)

st.markdown("--- ")

st.markdown("#### 👨‍💻 Equipe")
autores = [
    "Allan José Farias Ferreira",
    "João Pedro de Lima", 
    "Luan D Miranda Filizola Santos",
    "Márcia Alves de Assis Lima",
    "Mateus Nicolas Santos Lins"
]

col1, col2 = st.columns([3, 1])
with col1:
    for autor in autores:
        st.markdown(f"- {autor}")

with col2:
    st.markdown("🔗 [GitHub do projeto](https://github.com/allanjfarias/PISI3)")

st.markdown("---")
st.markdown("*💡 Desenvolvido para PISI3 - UFRPE*")
