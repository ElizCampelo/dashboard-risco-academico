import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import google.generativeai as genai

# --- Função Principal de Processamento e Treinamento ---

@st.cache_data
def processar_dados_e_treinar_modelo():
    """
    Função completa que carrega os dados, cria features, treina o modelo,
    e prepara os dados finais para o dashboard.
    """
    print("--- Início do Processamento e Treinamento ---")

    # --- Passo 1: Carregar os Dados ---
    try:
        df_modelagem = pd.read_csv('dados_alunos_ficticios.csv')
        print("Dados fictícios carregados com sucesso.")
    except FileNotFoundError:
        return None, 0 # Retorna None se o ficheiro não for encontrado

    # --- Passo 2: Engenharia de Features (Histórico do Aluno) ---
    print("A criar a feature de histórico do aluno...")
    df_modelagem['data_entrega'] = pd.to_datetime(df_modelagem['data_entrega'])
    df_modelagem.sort_values(by=['student_id', 'data_entrega'], inplace=True)
    df_modelagem['hist_correto_aluno'] = df_modelagem.groupby('student_id')['percentual_acertos'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    media_geral = df_modelagem['percentual_acertos'].mean()
    df_modelagem['hist_correto_aluno'].fillna(media_geral, inplace=True)
    print("Feature de histórico criada.")

    # --- Passo 3: Preparar Dados para o Modelo ---
    print("A preparar dados para o modelo...")
    colunas_categoricas = ['disciplina', 'instrumento']
    df_encoded = pd.get_dummies(df_modelagem, columns=colunas_categoricas, prefix=colunas_categoricas)

    y = df_encoded['percentual_acertos']
    X = df_encoded.drop(columns=[
        'percentual_acertos',
        'student_id',
        'student_name',
        'data_entrega'
    ])
    X = X.select_dtypes(include=np.number).fillna(0)
    print("Dados preparados.")

    # --- Passo 4: Treinar e Avaliar o Modelo ---
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    print("A iniciar o treinamento do modelo...")
    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_treino, y_treino)
    print("Treinamento concluído!")

    previsoes = modelo.predict(X_teste)
    erro = np.sqrt(mean_squared_error(y_teste, previsoes))
    erro_formatado = f"{erro:.2f}"
    print(f"\nO erro médio das previsões do modelo é de: {erro_formatado} pontos percentuais.")

    # --- PASSO 5: PREPARAÇÃO DOS DADOS PARA O DASHBOARD ---
    print("\n--- A preparar dados para o Dashboard ---")

    df_dashboard_source = df_modelagem.sort_values(by=['student_id', 'data_entrega'], ascending=[True, False])
    df_dashboard = df_dashboard_source.groupby('student_id').first().reset_index()

    colunas_dashboard = [
        'student_id',
        'student_name',
        'disciplina',
        'hist_correto_aluno',
        'data_entrega'
    ]
    df_dashboard_final = df_dashboard[colunas_dashboard]

    df_dashboard_final = df_dashboard_final.rename(columns={
        'hist_correto_aluno': 'indicador_desempenho',
        'data_entrega': 'data_ultima_atividade'
    })
    
    print("Dados para o dashboard preparados.")
    return df_dashboard_final, erro_formatado

# --- Interface Principal do Dashboard ---
st.set_page_config(layout="wide", page_title="Dashboard de Risco Acadêmico com IA")

# Executa a função de processamento para obter os dados
df, erro_modelo = processar_dados_e_treinar_modelo()

# Verifica se os dados foram carregados. Se não, exibe o erro e para.
if df is None:
    st.error("Erro: O ficheiro 'dados_alunos_ficticios.csv' não foi encontrado.")
    st.info("Por favor, execute primeiro o script 'gerador_dados_alunos.py' e coloque o CSV na mesma pasta que este aplicativo.")
    st.stop()
else:
    st.title("🎓 Dashboard de Risco Acadêmico com Assistente de IA")
    st.markdown("Analise o desempenho dos alunos no dashboard ou converse com a IA para obter insights.")
    st.info(f"**Performance do Modelo de IA:** O erro médio das previsões é de **{erro_modelo}** pontos percentuais.")

    # --- Barra Lateral (Sidebar) com os Filtros e API Key ---
    st.sidebar.header("Filtros e Configuração")
    
    api_key = st.sidebar.text_input("Sua Chave da API Gemini:", type="password", help="Insira sua chave da API do Google Gemini aqui.")
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.sidebar.error(f"Erro ao configurar a API: {e}")

    disciplinas = sorted(df['disciplina'].unique())
    disciplina_selecionada = st.sidebar.selectbox(
        "Filtrar por Última Disciplina:",
        options=['Todas'] + disciplinas
    )
    nome_aluno = st.sidebar.text_input("Buscar Aluno por Nome:")

    # --- Lógica de Filtragem dos Dados ---
    df_filtrado = df.copy()
    if disciplina_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['disciplina'] == disciplina_selecionada]
    if nome_aluno:
        df_filtrado = df_filtrado[df_filtrado['student_name'].str.contains(nome_aluno, case=False, na=False)]

    # --- Seção 1: Dashboard Principal ---
    st.header(f"📊 Análise de: {disciplina_selecionada}")
    
    LIMITE_ALTO_RISCO = 60
    LIMITE_RISCO_MODERADO = 75
    total_alunos = len(df_filtrado)
    alunos_alto_risco = len(df_filtrado[df_filtrado['indicador_desempenho'] < LIMITE_ALTO_RISCO])
    media_desempenho = df_filtrado['indicador_desempenho'].mean() if total_alunos > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Alunos", f"{total_alunos}")
    col2.metric(f"Alunos em Alto Risco (< {LIMITE_ALTO_RISCO}%)", f"{alunos_alto_risco}")
    col3.metric("Média de Desempenho", f"{media_desempenho:.2f}%")
    
    st.subheader("Situação Detalhada dos Alunos")
    def formatar_cor_desempenho(val):
        if val < LIMITE_ALTO_RISCO: cor = 'red'
        elif val < LIMITE_RISCO_MODERADO: cor = 'orange'
        else: cor = 'green'
        return f'color: {cor}; font-weight: bold;'

    df_para_exibir = df_filtrado[['student_name', 'disciplina', 'indicador_desempenho', 'data_ultima_atividade']] \
        .sort_values(by='indicador_desempenho').reset_index(drop=True) \
        .rename(columns={'student_name': 'Nome do Aluno', 'disciplina': 'Última Disciplina', 
                         'indicador_desempenho': 'Indicador de Desempenho (%)', 'data_ultima_atividade': 'Última Atividade'})

    st.dataframe(
        df_para_exibir.style.apply(lambda row: row.map(formatar_cor_desempenho), subset=['Indicador de Desempenho (%)']) \
            .format({'Indicador de Desempenho (%)': '{:.2f}', 'Última Atividade': '{:%d/%m/%Y}'}),
        use_container_width=True
    )

    st.divider()

    # --- Seção 2: Assistente de IA ---
    st.header("🤖 Converse com o Assistente de Dados")

    if not api_key:
        st.warning("Por favor, insira sua chave da API Gemini na barra lateral para ativar o assistente.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Faça uma pergunta sobre os dados filtrados..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            dados_contexto = df_filtrado.to_markdown(index=False)
            prompt_completo = f"""
            Você é um analista de dados educacionais. Sua função é analisar os dados fornecidos abaixo e responder à pergunta do usuário.
            Seja conciso e direto. Baseie sua resposta *estritamente* nos dados fornecidos.
            Se a pergunta não puder ser respondida com os dados, diga "Não tenho informações suficientes para responder a essa pergunta com os dados fornecidos."
            Não invente informações.

            **Dados Atuais (filtrados no dashboard):**
            {dados_contexto}

            **Pergunta do Usuário:**
            {prompt}
            """

            with st.chat_message("assistant"):
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt_completo)
                    resposta_ia = response.text
                    st.markdown(resposta_ia)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_ia})
                except Exception as e:

                    st.error(f"Ocorreu um erro ao contatar a IA: {e}")
