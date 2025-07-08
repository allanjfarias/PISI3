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
    df = pd.read_csv("datasets/spotify_processed.csv")
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
    'duration_ms': [-0.051071, 0.008570, 0.028559, 0.042813, -0.012008, 0.003364, -0.098845],
    'explicit': [-0.038581, 0.004771, -0.046049, 0.031375, 0.063666, -0.034393, -0.007478],
    'danceability': [-0.173577, 0.096539, 0.083526, 0.197486, 0.124737, -0.015987, -0.744664],
    'time_signature_4': [-0.121951, 0.360449, 0.359061, 0.359939, 0.225288, 0.360530, -2.751338],
    'time_signature_5': [0.008218, -0.136505, -0.134756, -0.135801, -0.057616, -0.136505, 1.036232],
    'is_popular': [0.122020, 0.112390, 0.094926, 0.119023, 0.132902, 0.104156, 0.079937]
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
