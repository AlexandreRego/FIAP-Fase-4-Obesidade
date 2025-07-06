import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# ======= Treinamento do Modelo (Pipeline completo) =======
@st.cache_resource
def treinar_pipeline():
    df = pd.read_csv("Obesity.csv")
    X = df.drop(columns=["Obesity"])
    y = df["Obesity"]

    numeric_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    categorical_features = ["Gender", "family_history", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# Treinar e salvar o modelo
modelo_pipeline = treinar_pipeline()
joblib.dump(modelo_pipeline, "random_forest_obesity_model.pkl")

# ======= Interface do App =======
st.set_page_config(page_title="Previsor de Obesidade", layout="wide")
st.title("🩺 Sistema Preditivo e Analítico de Obesidade")

st.markdown("Preencha os dados abaixo para obter uma previsão do nível de obesidade.")

# Entradas do usuário
with st.form("obesity_form"):
    gender = st.selectbox("Gênero", ["Male", "Female"])
    age = st.slider("Idade", 10, 100, 25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, step=0.01)
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, step=0.1)
    family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])
    favc = st.selectbox("Come alimentos muito calóricos com frequência?", ["yes", "no"])
    fcvc = st.slider("Frequência de vegetais na refeição (1-3)", 1.0, 3.0, 2.0)
    ncp = st.slider("Número de refeições principais por dia", 1.0, 4.0, 3.0)
    caec = st.selectbox("Petisca entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fuma?", ["yes", "no"])
    ch2o = st.slider("Quantidade de água por dia (litros)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitora calorias ingeridas?", ["yes", "no"])
    faf = st.slider("Frequência de atividade física (horas/semana)", 0.0, 5.0, 1.0)
    tue = st.slider("Tempo de tela diário (horas)", 0.0, 5.0, 1.0)
    calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Meio de transporte", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submitted = st.form_submit_button("Prever")

    if submitted:
        input_dict = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }
        input_df = pd.DataFrame([input_dict])
        pred = modelo_pipeline.predict(input_df)[0]
        st.success(f"Nível de obesidade previsto: **{pred}**")

# Seção de Insights Analíticos
st.markdown("---")
st.header("📊 Painel Analítico sobre Obesidade")

@st.cache_data
def load_data():
    df = pd.read_csv("Obesity.csv")
    return df

df = load_data()

st.subheader("Distribuição de Obesidade por Gênero")
graph_data = df.groupby(["Gender", "Obesity"]).size().unstack().fillna(0)
graph_data.T.plot(kind="bar", stacked=True, figsize=(10,5))
plt.xlabel("Nível de Obesidade")
plt.ylabel("Contagem")
st.pyplot(plt.gcf())

st.subheader("Média de Peso por Categoria de Obesidade")
peso_media = df.groupby("Obesity")["Weight"].mean().sort_values()
peso_media.plot(kind="barh")
plt.xlabel("Peso Médio (kg)")
plt.ylabel("Categoria de Obesidade")
st.pyplot(plt.gcf())

st.subheader("Matriz de Correlação entre Variáveis Numéricas")
corr = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt.gcf())
