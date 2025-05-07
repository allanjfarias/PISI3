import streamlit as st

st.set_page_config(
    page_title="Trabalho PISI3 - Grupo03",
    layout="wide",
    menu_items={
        'About': '''Trabalho de Machine Learning e Data Science para a disciplina Projeto Interdisciplinar em Sistemas de Informação III, do curso de Bacharelado em Sistemas de Informação da UFRPE.'''
    }
)

st.title("🎓 Dashboard - Grupo03")

st.markdown("""
Esta aplicação contém informações sobre o processo de **Data Science** e **Machine Learning** que serão aplicados ao dataset:

🎧 **900K+ Spotify Songs with Lyrics, Emotions & More**
""")

st.subheader("📄 Etapas do Projeto")
st.markdown("""
- 🔎 Exploratory Data Analysis
""")

st.markdown("🔗 [GitHub do Projeto](https://github.com/allanjfarias/PISI3)")

st.markdown("""---""")

st.markdown("#### 👨‍💻 Autores")
autores = [
    "Allan José Farias Ferreira",
    "João Pedro de Lima",
    "Luan D Miranda Filizola Santos",
    "Márcia Alves de Assis Lima",
    "Mateus Nicolas Santos Lins"
]
for autor in autores:
    st.markdown(f"- {autor}")
