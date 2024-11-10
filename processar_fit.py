import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
from fitparse import FitFile
from streamlit_folium import st_folium
import numpy as np

# Função para carregar e processar dados do arquivo .fit
def carregar_dados_fit(arquivo_fit):
    fitfile = FitFile(arquivo_fit)
    dados = []
    for record in fitfile.get_messages('record'):
        record_data = {}
        for field in record:
            record_data[field.name] = field.value
        dados.append(record_data)
    
    df = pd.DataFrame(dados)
    
    # Excluir linhas com valores None
    #df.dropna(inplace=True)
    
    # Calcula a velocidade
    df['speed_calculated'] = df['distance'].diff() / df['timestamp'].diff().dt.total_seconds()

    # Remove a primeira linha, que terá um valor NaN para a velocidade
    df = df.dropna()
    # Calcula a média móvel
    df['speed_smoothed'] = df['speed_calculated'].rolling(window=3, center=True).mean()
    
    # Substitui outliers pela média suavizada
    # Considera outlier se a diferença entre a velocidade original e a suavizada 
    # for maior que o desvio padrão da velocidade suavizada
    threshold = df['speed_smoothed'].std()
    df.loc[abs(df['speed_calculated'] - df['speed_smoothed']) > threshold, 'speed_calculated'] = df['speed_smoothed']  
    
    # Remove a coluna da média suavizada
    df = df.drop(columns=['speed_smoothed'])

    # Conversão de colunas com verificações adicionais
    df['speed_kmh'] = df['speed_calculated'] * 3.6  # Conversão para km/h

    # Calcula a média móvel da velocidade
    df['speed_smoothed'] = df['speed_kmh'].rolling(window=60, center=True).mean()
    
    # Interpola os valores NaN para criar uma linha contínua
    df['speed_smoothed'] = df['speed_smoothed'].interpolate()

    # Garantir que não haja valores negativos na coluna de 'speed_smoothed'
    if (df['speed_smoothed'] < 0).any():
        print("Aviso: Existem valores negativos em 'speed_smoothed'. Eles serão corrigidos para 0.")
        df.loc[df['speed_smoothed'] < 0, 'speed_smoothed'] = 0  # Corrigindo valores negativos

    # Calculando pace (min/km) com verificação para evitar divisão por zero
    df['pace_min_km'] = df['speed_smoothed'].apply(lambda x: 60 / x if x > 0 else None)  # Evita divisão por zero


    # Verificando se existem valores nulos após a interpolação
    if df['pace_min_km'].isnull().any():
        print("Aviso: Existem valores nulos em 'pace_min_km' após interpolação. Verifique os dados.")



    # Função para remover outliers com base no método IQR
    def remover_outliers_iqr(df, coluna):
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        return df[coluna].where(df[coluna].between(limite_inferior, limite_superior), np.nan)

    # Aplicando o método IQR nas colunas relevantes
    for col in ['speed_smoothed', 'altitude', 'heart_rate', 'cadence']:
        df[col] = remover_outliers_iqr(df, col)
    
    # Remover linhas com NaN após tratamento de outliers
    df.dropna(inplace=True)

    # Calculando inclinação média a cada quilômetro
    df['distancia_km'] = df['distance'] / 1000  # Converte distância para km
    df['inclination'] = df['altitude'].diff() / df['distancia_km'].diff()  # Calcula inclinação
    df['inclination'].fillna(0, inplace=True)  # Trata valores NaN resultantes
    
    return df

# Função para mostrar o percurso no mapa
def mostrar_mapa(df):
    if 'position_lat' in df.columns and 'position_long' in df.columns:
        mapa = folium.Map(location=[df['position_lat'].mean(), df['position_long'].mean()], zoom_start=13)
        coords = list(zip(df['position_lat'], df['position_long']))
        folium.PolyLine(coords, color="blue", weight=2.5, opacity=1).add_to(mapa)
        st_folium(mapa, width=700, height=500)
    else:
        st.write("Dados de localização não disponíveis no arquivo .fit.")

# Função para geração de gráficos aprimorados
def gerar_graficos(df):
    # Suavizando a linha com uma média móvel (janela de 5 elementos, ajustável)
    df['speed_kmh_smoothed'] = df['speed_smoothed'].rolling(window=30, min_periods=1).mean()

    # Plotando a velocidade suavizada
    st.subheader('Velocidade (km/h) ao longo do tempo')
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df['distancia_km'], df['speed_kmh_smoothed'], label='Velocidade Suavizada (km/h)', color='blue')
    ax.set_title('Velocidade ao longo do percurso')
    ax.set_xlabel('Distância')
    ax.set_ylabel('Velocidade (km/h)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Elevação e inclinação ao longo do tempo
    st.subheader('Elevação e Inclinação ao longo do percurso')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['timestamp'], df['altitude'], label='Elevação (m)', color='green')
    ax2 = ax.twinx()
    ax2.plot(df['timestamp'], df['inclination'], label='Inclinação (%)', color='orange')
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Elevação (m)', color='green')
    ax2.set_ylabel('Inclinação (%)', color='orange')
    fig.suptitle('Elevação e Inclinação ao longo do percurso')
    st.pyplot(fig)

    # Distribuição da Cadência (Boxplot)
    st.subheader('Distribuição da Cadência')
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.histplot(df['cadence'], kde=True, bins=30, color='purple', ax=ax)
    ax.set_title('Distribuição da Cadência')
    ax.set_xlabel('Cadência')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

    # Histograma da velocidade (km/h)
    st.subheader('Distribuição da Velocidade (km/h)')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['speed_kmh'], kde=True, bins=30, color='purple', ax=ax)
    ax.set_title('Distribuição da Velocidade (km/h)')
    ax.set_xlabel('Velocidade (km/h)')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

    # Gráfico de pace e batimentos cardíacos com zonas de intensidade
    st.subheader('Pace (min/km) e Batimentos Cardíacos')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['timestamp'], df['pace_min_km'], color='orange', label='Pace (min/km)')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Pace (min/km)', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')

    # Criando um segundo eixo y para batimentos cardíacos
    ax2 = ax1.twinx()
    ax2.plot(df['timestamp'], df['heart_rate'], color='red', label='Batimentos Cardíacos (bpm)')
    ax2.set_ylabel('Batimentos Cardíacos (bpm)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Adicionando zonas de intensidade no gráfico de batimentos cardíacos
    heart_rate_zones = [90, 120, 140, 160, 180]
    for zone in heart_rate_zones:
        ax2.axhline(y=zone, color='gray', linestyle='--', linewidth=0.5)
    
    fig.suptitle('Pace e Batimentos Cardíacos ao longo do tempo')
    st.pyplot(fig)

# Função principal
def main():
    st.title('Análise de Dados de Corrida Profissional')

    # Carregar o arquivo .fit
    arquivo_fit = st.file_uploader("Carregue seu arquivo .fit", type=['fit'])
    
    if arquivo_fit is not None:
        st.write("Arquivo carregado com sucesso!")
        
        # Carregar e processar dados
        df = carregar_dados_fit(arquivo_fit)

        # Exibir opções de visualização
        options = ["Visão Geral", "Gráficos", "Mapa", "Tabela Completa"]
        choice = st.sidebar.selectbox("Escolha uma opção", options)

        if choice == "Visão Geral":
            st.subheader("Resumo dos Dados")
            st.write(df.describe())
            st.subheader('Tabela de Dados')
            st.write(df.head())

        elif choice == "Gráficos":
            gerar_graficos(df)

        elif choice == "Mapa":
            st.subheader("Percurso no Mapa")
            mostrar_mapa(df)

        elif choice == "Tabela Completa":
            st.subheader("Tabela Completa de Dados")
            st.write(df)

if __name__ == '__main__':
    main()
