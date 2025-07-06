import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# =================================================================================
# CONFIGURAÇÃO DA PÁGINA
# =================================================================================
st.set_page_config(
    page_title="Análise de Obesidade | Dashboard Preditivo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================================
# FUNÇÕES DE CACHE E PROCESSAMENTO
# =================================================================================

@st.cache_resource
def carregar_e_treinar_modelo():
    """
    Carrega os dados, define o pipeline de pré-processamento e treina o modelo
    RandomForest. O resultado é cacheado para evitar re-treinamento a cada
    execução do script.
    """
    try:
        df_treino = pd.read_csv("Obesity.csv")
    except FileNotFoundError:
        st.error("Arquivo 'Obesity.csv' não encontrado. Por favor, certifique-se de que o arquivo está no diretório correto.")
        st.stop()
        
    X = df_treino.drop(columns=["Obesity"])
    y = df_treino["Obesity"]

    # Identificação automática das colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Definição dos transformadores para o pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Criação do pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Criação do pipeline final
    pipeline_obesidade = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline_obesidade.fit(X, y)
    return pipeline_obesidade

@st.cache_data
def carregar_dados_brutos():
    """Carrega o arquivo CSV bruto para análise e o armazena em cache."""
    try:
        dados = pd.read_csv("Obesity.csv")
        return dados
    except FileNotFoundError:
        return None

# --- Carregamento de Dados e Modelo ---
modelo_pipeline = carregar_e_treinar_modelo()
df = carregar_dados_brutos()

# =================================================================================
# INTERFACE PRINCIPAL DO STREAMLIT
# =================================================================================

# --- Título e Descrição ---
st.title("Análise de Obesidade | Dados para Insight's")
st.markdown("""
Esta aplicação utiliza Machine Learning para prever o nível de obesidade com base em características físicas e hábitos de vida.
**Use a barra lateral para inserir seus dados e obter uma previsão personalizada.** Explore o painel analítico abaixo para descobrir insights sobre os fatores que influenciam a obesidade.
""")

# --- Barra Lateral (Sidebar) para Entradas do Usuário ---
st.sidebar.header("⚙️ Insira seus Dados para Previsão")

with st.sidebar.form("formulario_previsao"):
    st.subheader("Informações Pessoais")
    gender = st.selectbox("Gênero", df['Gender'].unique(), index=0)
    age = st.slider("Idade", 14, 70, 25)
    height = st.slider("Altura (m)", 1.40, 2.10, 1.75)
    weight = st.slider("Peso (kg)", 40.0, 180.0, 80.0)

    st.subheader("Hábitos e Histórico Familiar")
    family_history = st.selectbox("Possui histórico familiar de sobrepeso?", ["yes", "no"], format_func=lambda x: "Sim" if x == 'yes' else "Não")
    favc = st.selectbox("Consome alimentos de alta caloria com frequência (FAVC)?", ["yes", "no"], format_func=lambda x: "Sim" if x == 'yes' else "Não")
    fcvc = st.slider("Frequência de consumo de vegetais (FCVC)", 1.0, 3.0, 2.0, help="1: Nunca, 2: Às vezes, 3: Sempre")
    ncp = st.slider("Número de refeições principais por dia", 1.0, 4.0, 3.0)
    caec = st.selectbox("Consome alimentos entre as refeições (CAEC)?", df['CAEC'].unique())
    smoke = st.selectbox("É fumante?", ["no", "yes"], format_func=lambda x: "Sim" if x == 'yes' else "Não")
    ch2o = st.slider("Consumo de água diário (Litros)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitora o consumo de calorias?", ["no", "yes"], format_func=lambda x: "Sim" if x == 'yes' else "Não")
    faf = st.slider("Frequência de atividade física (dias/semana)", 0.0, 7.0, 3.0)
    tue = st.slider("Tempo de uso de telas (horas/dia)", 0.0, 5.0, 1.0)
    calc = st.selectbox("Frequência de consumo de álcool (CALC)", df['CALC'].unique())
    mtrans = st.selectbox("Principal meio de transporte", df['MTRANS'].unique())

    botao_submeter = st.form_submit_button("Analisar e Prever")

# --- Exibição do Resultado da Previsão ---
if botao_submeter:
    input_dict = {
        "Gender": gender, "Age": age, "Height": height, "Weight": weight,
        "family_history": family_history, "FAVC": favc, "FCVC": fcvc, "NCP": ncp,
        "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o, "SCC": scc, "FAF": faf,
        "TUE": tue, "CALC": calc, "MTRANS": mtrans
    }
    input_df = pd.DataFrame([input_dict])
    
    predicao = modelo_pipeline.predict(input_df)[0]
    probabilidade_predicao = modelo_pipeline.predict_proba(input_df)
    imc = weight / (height ** 2)

    st.header("🎯 Resultado da Análise Preditiva")
    col_metrica1, col_metrica2 = st.columns(2)
    with col_metrica1:
        st.metric(label="Nível de Obesidade Previsto", value=predicao)
    with col_metrica2:
        st.metric(label="Seu IMC (Índice de Massa Corporal)", value=f"{imc:.2f}")

    st.info(f"Com base nos dados fornecidos, o modelo indica uma forte tendência para **{predicao}**.")
    
    st.subheader("Confiança da Previsão")
    df_prob = pd.DataFrame(probabilidade_predicao, columns=modelo_pipeline.classes_, index=["Probabilidade"]).T
    df_prob = df_prob.sort_values(by="Probabilidade", ascending=False)
    
    fig_prob = px.bar(
        df_prob, x=df_prob.index, y='Probabilidade',
        labels={'Probabilidade': 'Probabilidade', 'index': 'Nível de Obesidade'},
        text_auto='.2%', title="Probabilidade para Cada Categoria"
    )
    fig_prob.update_layout(yaxis_title="Probabilidade", xaxis_title="Nível de Obesidade")
    st.plotly_chart(fig_prob, use_container_width=True)

# --- Seção de Insights Analíticos ---
if df is not None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("📊 Painel Analítico Interativo")
    st.markdown("Explore as relações entre diferentes fatores e os níveis de obesidade no conjunto de dados.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição Geral da Obesidade")
        contagem_obesidade = df['Obesity'].value_counts()
        fig_pizza = px.pie(names=contagem_obesidade.index, values=contagem_obesidade.values,
                            title="Proporção de cada Nível de Obesidade", hole=0.3)
        fig_pizza.update_layout(legend_title_text='Níveis de Obesidade')
        st.plotly_chart(fig_pizza, use_container_width=True)

    with col2:
        st.subheader("Fatores Mais Relevantes na Previsão")
        importances = modelo_pipeline.named_steps['classifier'].feature_importances_
        # Acessando os nomes das features após o pré-processamento
        preprocessor = modelo_pipeline.named_steps['preprocessor']
        numeric_features = preprocessor.transformers_[0][2]
        cat_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(preprocessor.transformers_[1][2])
        all_features = numeric_features + list(cat_features_out)
        
        df_importancia = pd.DataFrame({'Fator': all_features, 'Importância': importances})
        df_importancia = df_importancia.sort_values(by='Importância', ascending=False).head(15)

        fig_importancia = px.bar(
            df_importancia, x='Importância', y='Fator', orientation='h',
            title='Importância de Cada Fator para o Modelo',
            labels={'Importância': 'Importância Relativa', 'Fator': 'Fator'},
            text='Importância'
        )
        fig_importancia.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_importancia.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_importancia, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribuição de Idade por Nível de Obesidade")
        fig_boxplot_idade = px.box(df, x="Obesity", y="Age", color="Obesity",
                                    title="Faixa Etária por Categoria de Obesidade",
                                    labels={"Obesity": "Nível de Obesidade", "Age": "Idade"})
        st.plotly_chart(fig_boxplot_idade, use_container_width=True)

    with col4:
        st.subheader("Distribuição de Obesidade por Gênero")
        obesidade_genero = df.groupby(['Gender', 'Obesity']).size().reset_index(name='Contagem')
        fig_barras_genero = px.bar(
            obesidade_genero, x="Obesity", y="Contagem", color="Gender", barmode="group",
            title="Contagem de Níveis de Obesidade por Gênero",
            labels={"Obesity": "Nível de Obesidade", "Contagem": "Número de Pessoas", "Gender": "Gênero"}
        )
        st.plotly_chart(fig_barras_genero, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Relação entre Peso, Altura e Nível de Obesidade")
    fig_dispersao = px.scatter(
        df, x="Height", y="Weight", color="Obesity",
        hover_data=['Age', 'Gender'], title="Dispersão de Peso vs. Altura",
        labels={"Height": "Altura (m)", "Weight": "Peso (kg)", "Obesity": "Nível de Obesidade"}
    )
    st.plotly_chart(fig_dispersao, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Matriz de Correlação entre Variáveis Numéricas")
    st.markdown("Esta matriz mostra como as variáveis numéricas se relacionam. Valores próximos de 1 (vermelho) ou -1 (azul) indicam forte correlação.")
    df_numerico = df.select_dtypes(include=[np.number])
    correlacao = df_numerico.corr()

    fig_heatmap = px.imshow(
        correlacao, text_auto=True, aspect="auto",
        color_continuous_scale='RdBu_r', title="Mapa de Calor das Correlações"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)