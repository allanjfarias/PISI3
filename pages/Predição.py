import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Previsão", layout="wide")
st.title("🤖 Previsão de popularidade")

@st.cache_resource
def load_models():
    models = {}
    try:
        models["Random Forest"] = joblib.load("modelos/random_forest_smote_pipeline.joblib")
        models["XGBoost"] = joblib.load("modelos/xgboost_smote_pipeline.joblib")
        models["SVM"] = joblib.load("modelos/svm_pipeline.joblib")
        models["KNN"] = joblib.load("modelos/knn_smote_pipeline_model.joblib")
        return models
    except FileNotFoundError as e:
        st.error(f"❌ Modelo não encontrado: {e.filename}. Certifique-se de que os arquivos .joblib estão na pasta 'modelos/'.")
        return None

@st.cache_data
def load_example_data():
    try:
        df = pd.read_csv("datasets_paralelos/kmeans_7clusters.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Arquivo 'datasets_paralelos/kmeans_7clusters.csv' não encontrado. Ele é necessário para preencher os valores iniciais dos seletores.")
        return None

models = load_models()
example_data = load_example_data()

if models is None or example_data is None:
    st.warning("A aplicação não pode continuar sem os modelos e o arquivo de dados de exemplo.")
    st.stop()

st.success("✅ Modelos e dados carregados com sucesso!")

st.sidebar.header("⚙️ Seleção de Modelo")
model_name = st.sidebar.selectbox("Escolha o modelo para a predição:", list(models.keys()))
model = models[model_name]

try:
    expected_features = model.feature_names_in_
except AttributeError:
    st.error(
        f"❌ O modelo '{model_name}' não contém a lista de features ('feature_names_in_'). "
        f"Ele pode não ter sido treinado com um DataFrame do Pandas. "
        f"Por favor, retreine o modelo garantindo que os nomes das colunas sejam salvos."
    )
    st.stop()

with st.expander("ℹ️ Informações do modelo"):
    st.write(f"**Tipo de modelo:** {model_name}")
    st.write(f"**Número de características esperadas pelo modelo:** {len(expected_features)}")
    st.write("**Classes de predição:** Não Popular, Popular")
    st.write("**Features esperadas:**")
    st.code(f"{list(expected_features)}", language="python")

st.header("🎯 Fazer predição")
st.write("Ajuste os parâmetros abaixo para prever se uma música será popular:")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎵 Características básicas")
    loudness = st.slider(
        "Volume (normalizado)",
        float(example_data["loudness"].min()),
        float(example_data["loudness"].max()),
        float(example_data["loudness"].mean())
    )
    tempo = st.slider(
        "Tempo (normalizado)",
        float(example_data["tempo"].min()),
        float(example_data["tempo"].max()),
        float(example_data["tempo"].mean())
    )

with col2:
    st.subheader("🎼 Características musicais")
    danceability = st.slider(
        "Dançabilidade",
        float(example_data["danceability"].min()),
        float(example_data["danceability"].max()),
        float(example_data["danceability"].mean())
    )
    energy = st.slider(
        "Energia",
        float(example_data["energy"].min()),
        float(example_data["energy"].max()),
        float(example_data["energy"].mean())
    )
    valence = st.slider(
        "Valência (positividade)",
        float(example_data["valence"].min()),
        float(example_data["valence"].max()),
        float(example_data["valence"].mean())
    )
    acousticness = st.slider(
        "Acústico",
        float(example_data["acousticness"].min()),
        float(example_data["acousticness"].max()),
        float(example_data["acousticness"].mean())
    )

with col3:
    st.subheader("🎤 Características avançadas")
    speechiness = st.slider(
        "Fala",
        float(example_data["speechiness"].min()),
        float(example_data["speechiness"].max()),
        float(example_data["speechiness"].mean())
    )
    instrumentalness = st.slider(
        "Instrumental",
        float(example_data["instrumentalness"].min()),
        float(example_data["instrumentalness"].max()),
        float(example_data["instrumentalness"].mean())
    )
    liveness = st.slider(
        "Ao vivo",
        float(example_data["liveness"].min()),
        float(example_data["liveness"].max()),
        float(example_data["liveness"].mean())
    )

def prepare_input_data(model_features):
    ui_inputs = {
        "loudness": loudness,
        "tempo": tempo,
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
    }

    input_dict = {feature: 0.0 for feature in model_features}
    for feature, value in ui_inputs.items():
        if feature in input_dict:
            input_dict[feature] = value

    return pd.DataFrame([input_dict])[model_features]

def predict_proba_wrapper(X_array):
    X_df = pd.DataFrame(X_array, columns=expected_features)
    return model.predict_proba(X_df)

if st.button("🚀 Fazer predição", type="primary"):
    input_df = prepare_input_data(expected_features)

    st.write("---")
    st.header("📊 Resultado da Predição")

    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        col1_res, col2_res = st.columns(2)

        with col1_res:
            if prediction == 1:
                st.success("🎉 **POPULAR** - Esta música tem potencial para ser popular!")
            else:
                st.info("📉 **NÃO POPULAR** - Esta música provavelmente não será muito popular.")
            st.metric("Confiança da Predição (Classe Predita)", f"{max(prediction_proba):.1%}")

        st.subheader(f"🔍 Explicação da Predição com SHAP ({model_name})")

        spinner_message = f"Calculando valores SHAP para {model_name}..."
        if model_name in ["KNN", "SVM"]:
            spinner_message += " Isso pode levar alguns segundos."

        with st.spinner(spinner_message):
            if hasattr(model, 'steps'):
                model_for_shap = model.steps[-1][1]
            else:
                model_for_shap = model

            if model_name == "XGBoost":
                explainer = shap.TreeExplainer(model_for_shap)
                shap_values_raw = explainer.shap_values(input_df)
                
                if isinstance(shap_values_raw, list):
                    shap_values = np.array(shap_values_raw[prediction]).flatten()
                else:
                    shap_values = np.array(shap_values_raw).flatten()
                
                if isinstance(explainer.expected_value, list):
                    base_value = explainer.expected_value[prediction]
                else:
                    base_value = explainer.expected_value

            elif model_name == "Random Forest":
                explainer = shap.TreeExplainer(model_for_shap)
                shap_values_raw = explainer.shap_values(input_df)

                if isinstance(shap_values_raw, list):
                    shap_values = np.array(shap_values_raw[prediction]).flatten()
                else:
                    shap_values = np.array(shap_values_raw).flatten()

                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    base_value = explainer.expected_value[prediction] 
                else:
                    base_value = explainer.expected_value

            elif model_name in ["SVM", "KNN"]:
                X_train_summary = shap.kmeans(example_data[expected_features].values, 25)
                explainer = shap.KernelExplainer(predict_proba_wrapper, X_train_summary)
                shap_values_all = explainer.shap_values(input_df.values, nsamples=100)

                if isinstance(shap_values_all, list):
                    shap_values = shap_values_all[prediction][0]
                else:
                    shap_values = shap_values_all[0, :, prediction]

                base_value = explainer.expected_value[prediction]

            if len(shap_values) != len(expected_features):
                corrected_shap = np.zeros(len(expected_features))
                min_length = min(len(shap_values), len(expected_features))
                corrected_shap[:min_length] = shap_values[:min_length]
                shap_values = corrected_shap

            contrib_df = pd.DataFrame({
                "Feature": expected_features,
                "SHAP Value": shap_values
            })
            contrib_df["Efeito"] = np.where(contrib_df["SHAP Value"] > 0, "Aumenta 📈", "Diminui 📉")
            contrib_df["abs_shap"] = np.abs(contrib_df["SHAP Value"])
            contrib_df = contrib_df.sort_values("abs_shap", ascending=False).head(15)

            fig_shap = px.bar(
                contrib_df.sort_values("abs_shap", ascending=True),
                x="SHAP Value",
                y="Feature",
                color="Efeito",
                color_discrete_map={"Aumenta 📈": "#2ca02c", "Diminui 📉": "#d62728"},
                title="Principais Features que Influenciaram a Predição",
                labels={"SHAP Value": "Contribuição SHAP para a Popularidade"},
                orientation="h"
            )
            fig_shap.update_layout(height=500, yaxis_title=None)

            st.info(
                f"""**Como interpretar:** O gráfico mostra o quanto cada característica "empurrou" a previsão final a partir de uma previsão base ({base_value:.3f}).
                Características em verde aumentaram a chance da música ser popular, e as em vermelho diminuíram."""
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        st.subheader("Cascata SHAP (Waterfall Plot)")
        st.info(
            """**Como interpretar:** O gráfico de cascata mostra como cada característica contribui para a previsão final, começando da previsão base e adicionando ou subtraindo o impacto de cada característica até chegar à previsão da música."""
        )

        contrib_df = contrib_df.sort_values("abs_shap", ascending=False).head(15)
        contrib_df = contrib_df[::-1]

        measure = ["relative"] * len(contrib_df)
        x = contrib_df["Feature"].tolist()
        y = contrib_df["SHAP Value"].tolist()
        text = [f"{v:+.3f}" for v in y]

        fig_waterfall = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measure,
            x=x,
            text=text,
            textposition="outside",
            y=y,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#d62728"}},
            increasing={"marker": {"color": "#2ca02c"}}
        ))

        fig_waterfall.update_layout(
            title="Impacto das Features na Predição (SHAP Waterfall)",
            yaxis_title="Contribuição SHAP",
            xaxis_title="Features",
            showlegend=False,
            height=600
        )

        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.subheader("Mapa de Calor das Contribuições SHAP")

        heatmap_df = contrib_df.sort_values("abs_shap", ascending=True).set_index("Feature")[["SHAP Value"]]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_df["SHAP Value"].values.reshape(-1, 1),
            y=heatmap_df.index.tolist(),
            x=["Contribuição SHAP"],
            colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
            colorbar=dict(title="Valor SHAP"),
            zmid=0
        ))

        fig_heatmap.update_layout(
            height=400,
            yaxis=dict(autorange="reversed"),
            xaxis_title="",
            yaxis_title="Feature",
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    except Exception as e:
        st.error(f"Ocorreu um erro durante a predição ou explicação: {e}")
        st.exception(e)
        st.error("Verifique se os inputs do modelo estão corretos e se o modelo carregado é válido.")