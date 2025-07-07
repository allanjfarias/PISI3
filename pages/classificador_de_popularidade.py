import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import os
import matplotlib.pyplot as plt

# Configurações iniciais da página do Streamlit
st.set_page_config(
    page_title="Classificador de Popularidade",
    page_icon="🎵",
    layout="wide"
)

# Armazena os arquivos em cache para que não sejam recarregados a cada interação do usuário.
@st.cache_resource
def load_files(pipeline_path, background_path):
    """Carrega o pipeline do modelo e os dados de background para o SHAP."""
    try:
        with open(pipeline_path, 'rb') as f_model:
            model_pipeline = pickle.load(f_model)
        background_data = pd.read_csv(background_path)
        return model_pipeline, background_data
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None

# Constrói os caminhos para os arquivos de forma relativa à localização do script.
pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'modelos', 'pipeline_completo.pkl')
background_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'X_train_background.csv')

# Carrega os artefatos necessários para a aplicação.
modelo_pipeline, X_train_background = load_files(pipeline_path, background_path)

# --- Interface do Usuário na Barra Lateral ---
st.sidebar.header("Insira as Características da Música")

# Verifica se os arquivos foram carregados antes de continuar.
if not all([modelo_pipeline, X_train_background is not None]):
    st.error("Arquivos essenciais não foram encontrados. Verifique as pastas 'modelos' e 'datasets'.")
    st.stop()

# Agrupa os inputs em um formulário para que a página só seja atualizada ao clicar no botão.
with st.sidebar.form(key='prediction_form'):
    st.write("Ajuste os valores para corresponder à música que deseja analisar.")
    
    # Cria os campos de entrada para cada uma das 10 features do modelo.
    danceability = st.slider('Dançabilidade', 0.0, 1.0, 0.7, 0.01)
    energy = st.slider('Energia', 0.0, 1.0, 0.8, 0.01)
    loudness = st.slider('Volume (Loudness)', -60.0, 0.0, -5.5, 0.5)
    speechiness = st.slider('Vocalização', 0.0, 1.0, 0.1, 0.01)
    acousticness = st.slider('Acústica', 0.0, 1.0, 0.2, 0.01)
    instrumentalness = st.slider('Instrumentalidade', 0.0, 1.0, 0.0, 0.01)
    liveness = st.slider('Ao Vivo', 0.0, 1.0, 0.15, 0.01)
    valence = st.slider('Positividade', 0.0, 1.0, 0.5, 0.01)
    tempo = st.number_input('Andamento (BPM)', min_value=0, max_value=250, value=120)
    duration_ms = st.number_input('Duração (ms)', min_value=0, value=220000, step=1000)
    
    submit_button = st.form_submit_button(label='Fazer Previsão')

# --- Lógica Principal da Página ---
st.title("Classificador de Popularidade de Músicas")
st.markdown("Use a barra lateral para inserir os dados da música e clique em 'Fazer Previsão' para ver o resultado.")

if submit_button:
    # Garante que a ordem das colunas do input seja a mesma que o modelo espera.
    colunas_do_modelo = X_train_background.columns.tolist()
    
    input_data = pd.DataFrame({
        'instrumentalness': [instrumentalness], 'acousticness': [acousticness], 'duration_ms': [duration_ms],
        'valence': [valence], 'energy': [energy], 'loudness': [loudness], 'liveness': [liveness],
        'tempo': [tempo], 'danceability': [danceability], 'speechiness': [speechiness]
    })
    input_data = input_data[colunas_do_modelo]
    
    # Converte todos os dados para float64 para evitar instabilidade numérica no scaler.
    input_data = input_data.astype(np.float64)

    # O pipeline salvo cuida internamente do escalonamento e da previsão.
    prediction = modelo_pipeline.predict(input_data)[0]
    prediction_proba = modelo_pipeline.predict_proba(input_data)[0]

    st.subheader("Resultado da Previsão")
    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.success("Potencial para ser Popular!")
    else:
        col1.error("Baixo potencial para ser Popular.")
    col2.metric(label="Confiança do Modelo (Popular)", value=f"{prediction_proba[1]:.2%}")
    st.divider()

    # --- Explicação da Previsão com SHAP ---
    st.subheader("Análise da Decisão do Modelo (SHAP)")
    
    try:
        # Acessa os componentes (scaler e classifier) de dentro do pipeline salvo.
        scaler_from_pipeline = modelo_pipeline.named_steps['scaler']
        classifier_from_pipeline = modelo_pipeline.named_steps['classifier']

        # Escala os dados de entrada manualmente, pois o explainer do SHAP precisa dos dados transformados.
        input_data_scaled = scaler_from_pipeline.transform(input_data)
        input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=colunas_do_modelo)
        
        # Cria o explainer para modelos de árvore e calcula os valores SHAP.
        explainer = shap.TreeExplainer(classifier_from_pipeline, X_train_background)
        shap_values = explainer.shap_values(input_data_scaled_df)

        st.write("O gráfico abaixo mostra quais características **aumentaram** (em vermelho) ou **diminuíram** (em azul) a chance da música ser classificada como 'Popular'.")
        
        fig, ax = plt.subplots()
        # Gera o gráfico de força do SHAP, mostrando os valores originais para melhor interpretação.
        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0,:],
            features=input_data.iloc[0,:],
            matplotlib=True, show=False
        )
        st.pyplot(fig, bbox_inches='tight')
        plt.clf()
    except Exception as e:
        st.error(f"Ocorreu um erro ao gerar o gráfico SHAP: {e}")
