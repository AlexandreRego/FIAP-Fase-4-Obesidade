import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise | Previsão Obesidade",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções de Cache ---

@st.cache_resource
def treinar_pipeline():
    """
    Carrega os dados, define o pré-processamento e treina um modelo RandomForest.
    Usa @st.cache_resource para que o modelo seja treinado apenas uma vez.
    """
    try:
        df = pd.read_csv("Obesity.csv")
    except FileNotFoundError:
        st.error("Arquivo 'Obesity.csv' não encontrado. Por favor, certifique-se de que o arquivo está no diretório correto.")
        st.stop()
        
    X = df.drop(columns=["Obesity"])
    y = df["Obesity"]

    # Define as colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Cria os transformadores para as colunas
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Cria o pré-processador com ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Cria o pipeline final com pré-processador e classificador
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

@st.cache_data
def load_data():
    """Carrega o arquivo CSV em um DataFrame e armazena em cache."""
    try:
        df = pd.read_csv("Obesity.csv")
        return df
    except FileNotFoundError:
        return None

# --- Carregamento de Dados e Modelo ---

# Carrega o pipeline treinado (treina na primeira execução)
modelo_pipeline = treinar_pipeline()
# Salvar o modelo treinado (opcional, mas bom para reuso)
joblib.dump(modelo_pipeline, "random_forest_obesity_model.pkl")

# Carrega os dados brutos para a análise
df = load_data()

if df is None:
    st.warning("Não foi possível carregar os dados para a análise visual.")
    st.stop()


# --- Interface do App ---

# Título e descrição principal
st.title("🩺 Sistema Preditivo e Analítico de Obesidade")
st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning para prever o nível de obesidade com base em hábitos de vida e características físicas.
**Use a barra lateral para inserir os dados e obter uma previsão.** Explore o painel analítico abaixo para entender melhor os fatores relacionados à obesidade.
""")

# --- Barra Lateral (Sidebar) para Entradas do Usuário ---
st.sidebar.header("Faça sua Previsão Aqui")

with st.sidebar.form("obesity_form"):
    st.subheader("Informações Pessoais")
    gender = st.selectbox("Gênero", df['Gender'].unique(), index=0)
    age = st.slider("Idade", 14, 65, 25)
    height = st.slider("Altura (m)", 1.40, 2.10, 1.75)
    weight = st.slider("Peso (kg)", 40.0, 180.0, 80.0)

    st.subheader("Hábitos e Histórico")
    family_history = st.selectbox("Histórico familiar de sobrepeso?", ["yes", "no"])
    favc = st.selectbox("Come alimentos calóricos com frequência (FAVC)?", ["yes", "no"])
    fcvc = st.slider("Frequência de consumo de vegetais (FCVC)", 1.0, 3.0, 2.0, help="1: Nunca, 2: Às vezes, 3: Sempre")
    ncp = st.slider("Número de refeições principais por dia", 1.0, 4.0, 3.0)
    caec = st.selectbox("Come algo entre as refeições (CAEC)?", df['CAEC'].unique())
    smoke = st.selectbox("Fuma?", ["no", "yes"])
    ch2o = st.slider("Consumo de água diário (Litros)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitora as calorias que ingere?", ["no", "yes"])
    faf = st.slider("Frequência de atividade física (dias/semana)", 0.0, 7.0, 3.0)
    tue = st.slider("Tempo de uso de tecnologia (horas/dia)", 0.0, 5.0, 1.0)
    calc = st.selectbox("Consumo de álcool (CALC)", df['CALC'].unique())
    mtrans = st.selectbox("Meio de transporte principal", df['MTRANS'].unique())

    submitted = st.form_submit_button("Analisar e Prever")

# --- Exibição do Resultado da Previsão ---
if submitted:
    input_dict = {
        "Gender": gender, "Age": age, "Height": height, "Weight": weight,
        "family_history": family_history, "FAVC": favc, "FCVC": fcvc, "NCP": ncp,
        "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o, "SCC": scc, "FAF": faf,
        "TUE": tue, "CALC": calc, "MTRANS": mtrans
    }
    input_df = pd.DataFrame([input_dict])
    
    # Faz a previsão e obtém as probabilidades
    pred = modelo_pipeline.predict(input_df)[0]
    pred_proba = modelo_pipeline.predict_proba(input_df)
    
    # Calcula o IMC para dar mais contexto
    imc = weight / (height ** 2)

    st.header(f"🔎 Resultado da Análise")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Nível de Obesidade Previsto", value=pred)
    with col2:
        st.metric(label="Seu IMC Calculado", value=f"{imc:.2f}")

    st.info(f"Com base nos dados fornecidos, o modelo prevê um nível de **{pred}**.")
    
    # Mostra a probabilidade da previsão em um gráfico de barras
    st.subheader("Confiança da Previsão")
    prob_df = pd.DataFrame(pred_proba, columns=modelo_pipeline.classes_, index=["Probabilidade"]).T
    prob_df = prob_df.sort_values(by="Probabilidade", ascending=False)
    
    fig_prob = px.bar(
        prob_df, 
        x=prob_df.index, 
        y='Probabilidade',
        labels={'Probabilidade': 'Probabilidade', 'index': 'Nível de Obesidade'},
        text_auto='.2%',
        title="Probabilidade para Cada Categoria"
    )
    fig_prob.update_layout(yaxis_title="Probabilidade", xaxis_title="Nível de Obesidade")
    st.plotly_chart(fig_prob, use_container_width=True)


# --- Seção de Insights Analíticos ---

st.markdown("<hr>", unsafe_allow_html=True)
st.header("📊 Painel Analítico Interativo")
st.markdown("Explore as relações entre diferentes fatores e os níveis de obesidade no conjunto de dados.")

# --- Layout em colunas para os gráficos ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição Geral da Obesidade")
    obesity_counts = df['Obesity'].value_counts()
    fig_pie = px.pie(
        names=obesity_counts.index, 
        values=obesity_counts.values,
        title="Proporção de cada Nível de Obesidade",
        hole=0.3
    )
    fig_pie.update_layout(legend_title_text='Níveis de Obesidade')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Fatores Mais Importantes na Previsão")
    # Extraindo as importâncias do modelo
    importances = modelo_pipeline.named_steps['classifier'].feature_importances_
    # Recuperando os nomes das features após o pré-processamento
    cat_features_out = modelo_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(modelo_pipeline.named_steps['preprocessor'].transformers_[1][2])
    numeric_features = modelo_pipeline.named_steps['preprocessor'].transformers_[0][2]
    all_features = numeric_features + list(cat_features_out)
    
    feature_importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)

    fig_importance = px.bar(
        feature_importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Importância de Cada Fator para o Modelo',
        labels={'importance': 'Importância Relativa', 'feature': 'Fator'},
        text='importance'
    )
    fig_importance.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribuição de Idade por Nível de Obesidade")
    fig_age_box = px.box(
        df,
        x="Obesity",
        y="Age",
        color="Obesity",
        title="Idade por Categoria de Obesidade",
        labels={"Obesity": "Nível de Obesidade", "Age": "Idade"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col4:
    st.subheader("Distribuição de Obesidade por Gênero")
    gender_obesity = df.groupby(['Gender', 'Obesity']).size().reset_index(name='count')
    fig_gender_bar = px.bar(
        gender_obesity,
        x="Obesity",
        y="count",
        color="Gender",
        barmode="group",
        title="Contagem de Obesidade por Gênero",
        labels={"Obesity": "Nível de Obesidade", "count": "Número de Pessoas", "Gender": "Gênero"}
    )
    st.plotly_chart(fig_gender_bar, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("Relação entre Peso, Altura e Obesidade")
fig_scatter = px.scatter(
    df,
    x="Height",
    y="Weight",
    color="Obesity",
    hover_data=['Age', 'Gender'],
    title="Peso vs. Altura por Nível de Obesidade",
    labels={"Height": "Altura (m)", "Weight": "Peso (kg)", "Obesity": "Nível de Obesidade"},
    color_discrete_sequence=px.colors.qualitative.Plotly
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("Matriz de Correlação entre Variáveis Numéricas")
st.markdown("Esta matriz mostra como as variáveis numéricas se relacionam entre si. Valores próximos de 1 (vermelho) ou -1 (azul) indicam uma forte correlação.")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig_heatmap = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title="Correlação entre as Características Numéricas"
)
st.plotly_chart(fig_heatmap, use_container_width=True)
