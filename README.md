# dashboard-risco-academico


📊 Dashboard de Risco Académico com Assistente de IA
Este projeto é uma aplicação web interativa construída com Streamlit que utiliza um modelo de Machine Learning para prever o desempenho de alunos e identificar aqueles em risco académico. Além do dashboard visual, a aplicação conta com um assistente de IA integrado (usando a API do Google Gemini) que permite aos utilizadores fazer perguntas em linguagem natural sobre os dados apresentados.

Funcionalidades Principais
Modelo Preditivo: Um modelo de RandomForestRegressor é treinado em tempo real para prever o desempenho dos alunos com base no seu histórico.

Dashboard Interativo: Visualização clara da situação dos alunos, com métricas chave e indicadores de risco coloridos (verde, laranja, vermelho).

Filtros Dinâmicos: Permite filtrar os dados por  disciplina e procurar por alunos específicos.

Assistente de IA: Uma interface de chat onde os educadores podem "conversar" com os dados, pedindo resumos, identificando alunos específicos e obtendo insights rápidos sem precisar de manipular tabelas.

🚀 Tecnologias Utilizadas
Python: Linguagem principal do projeto.

Streamlit: Framework para a construção da aplicação web e do dashboard.

Pandas: Para manipulação e preparação dos dados.

Scikit-learn: Para a criação e treinamento do modelo de Machine Learning.

Google Generative AI (Gemini): Para alimentar o assistente de IA conversacional.

🌐 Deploy da Aplicação
A aplicação está disponível online! Acesse através do link abaixo:

https://dashboard-risco-academico-dfz67vmgj5pzbtaa62zybn.streamlit.app/

#Estrutura do Projeto
main.py: O ficheiro principal da aplicação Streamlit que contém toda a lógica do dashboard e da IA.

gerador_dados_alunos.py: Script para gerar a base de dados fictícia (dados_alunos_ficticios.csv).

dados_alunos_ficticios.csv: A base de dados utilizada pela aplicação.

requirements.txt: Lista de todas as dependências Python necessárias para o projeto.

README.md: Este ficheiro.
