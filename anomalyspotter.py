# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import seaborn as sns

# Configurações do Streamlit
st.set_page_config(layout="wide",
	page_title= "AnomalySpotter",
	page_icon="https://i.postimg.cc/Hkj2GxMk/icon-app.png")
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True) 

# Função para detectar máquinas problemáticas usando Isolation Forest
def detect_anomalies_isolation_forest(data):
    model = IsolationForest(contamination=0.2)
    preds = model.fit_predict(data)
    return np.where(preds == -1)[0]

# Função para detectar máquinas problemáticas usando PCA e distância de Mahalanobis
def detect_anomalies_pca(data):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    cov_matrix = np.cov(data_pca, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_distances = pairwise_distances(data_pca, metric="mahalanobis", VI=inv_cov_matrix)
    threshold = np.percentile(mahalanobis_distances, 95)
    anomalies = np.where(mahalanobis_distances > threshold)[0]
    return anomalies

# Função para detectar máquinas problemáticas usando DBSCAN
def detect_anomalies_dbscan(data):
    dbscan = DBSCAN(eps=1.5, min_samples=3)
    preds = dbscan.fit_predict(data)
    return np.where(preds == -1)[0]

# Função principal do Streamlit
def main():
    st.image("https://i.postimg.cc/85gLQvnS/logo-app.png",width=380)
    # Utilizando HTML para personalizar a exibição do rodapé
    st.markdown(
        """
        <style>
            .rodape {
                background : #000;
                width: 100%;
                font-size: 14px;
                color: #fff;
                text-align: justify;
                height:24px
            }
        </style>
        """, unsafe_allow_html=True
    )
    # Usando a classe .rodape para exibir o texto com a formatação desejada
    st.markdown("<div class='rodape'>&nbsp;&nbsp;<b>Versão</b>:&nbsp; 1.0&nbsp;&nbsp;|&nbsp;&nbsp;<b>Streamlit</b>&nbsp;&nbsp;|&nbsp;&nbsp;<b>Desenvolvido por</b>&nbsp:&nbsp;Onédio S Seabra Junior</div>", unsafe_allow_html=True)

    # Upload do dataset
    st.header("Upload da Base de Dados")
    uploaded_file = st.file_uploader("Carregue seu conjunto de dados (formato CSV)", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.info('Base de dados carregadas com sucesso!')
        
        #Carrega informações sobre a base de dados
        st.header("Informações da Base de Dados")
        col1, col2, col3 = st.columns(3)
        col1.metric("Quantidade de Máquinas", f"{data.shape[0]}")
        col2.metric("Quantidade de Sensores", f"{data.shape[1]-1}")
        col3.metric("Amostras Carregadas", "5")
        st.subheader("Amostra Carregada")
        st.write(data.sample(5))

        # Normalização dos dados
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(data.drop('Seq', axis=1))

        st.header("Modelos")
        tab1, tab2, tab3 = st.tabs(["DETECÇÃO DE OUTLIERS USANDO ISOLATION FOREST", "PCA SEGUIDO DE DETECÇÃO DE DISTÂNCIAS ANÔMALAS", "DBSCAN"])
        # Aba para Isolation Forest
        with tab1:
            st.write("A :blue[DETECÇÃO DE OUTLIERS] refere-se ao processo de reconhecimento de pontos de dados que apresentam um desvio notável em relação ao conjunto geral de dados. No contexto apresentado, se uma máquina exibe leituras de sensores significativamente distintas das outras, ela pode estar apresentando falhas ou irregularidades.")
            st.write("Para realizar essa tarefa, empregamos o método [Isolation Forest]. Esse algoritmo, fundamentado em estruturas de árvores, trabalha isolando as observações, diferenciando-as do restante do conjunto.")
            anomaly_indices_forest = detect_anomalies_isolation_forest(X_normalized)
            anomaly_machines_forest = data.iloc[anomaly_indices_forest]['Seq'].tolist()
            st.warning(f'Máquina(s) defeituosa(s) : {anomaly_machines_forest}', icon="⚠️")
            
        # Aba para PCA
        with tab2:
            st.write("O :blue[PCA]  (Análise de Componentes Principais) é uma técnica utilizada para simplificar a complexidade dos dados, reduzindo suas dimensões.")
            st.write("Depois de aplicar o PCA e condensar as dimensões dos dados, o próximo passo é determinar a distância de Mahalanobis para cada ponto no novo espaço simplificado. Pontos que apresentam distâncias excepcionalmente elevadas são categorizados como atípicos ou anômalos.")
            anomaly_indices_pca = detect_anomalies_pca(X_normalized)
            anomaly_machines_pca = data.iloc[anomaly_indices_pca]['Seq'].tolist()
            st.warning(f'Máquina(s) defeituosa(s) : {list(set(anomaly_machines_pca))}', icon="⚠️")
            
        # Aba para DBSCAN
        with tab3:
            st.write("O :blueé uma técnica de agrupamento que identifica grupos com diferentes densidades e é capaz de reconhecer pontos considerados como ruído ou anômalos.")
            st.write("A lógica central do DBSCAN é que um ponto anômalo é aquele que não pertence a um grupo denso de dados. Assim, o algoritmo agrupa pontos que estão próximos entre si, enquanto pontos isolados em áreas menos densas são classificados como anomalias.")
            anomaly_indices_dbscan = detect_anomalies_dbscan(X_normalized)
            anomaly_machines_dbscan = data.iloc[anomaly_indices_dbscan]['Seq'].tolist()
            st.warning(f'Máquina(s) defeituosa(s) : {anomaly_machines_dbscan}', icon="⚠️")

        # Avaliação majoritária
        st.header("Máquina(s) Defeituosa(s)")
        all_anomalies = np.concatenate([anomaly_indices_forest, anomaly_indices_pca, anomaly_indices_dbscan])
        unique, counts = np.unique(all_anomalies, return_counts=True)
        majority_vote_anomalies = unique[counts >= 3]
        majority_vote_machines = data.iloc[majority_vote_anomalies]['Seq'].tolist()
        st.error(f'Máquina(s) defeituosa(s) : {majority_vote_machines}', icon="🚨")
                

if __name__ == "__main__":
    main()