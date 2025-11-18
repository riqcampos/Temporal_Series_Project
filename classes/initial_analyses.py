import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings

# Configurações visuais para apresentações profissionais
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class TimeSeriesExplorer:
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str, freq: str = None):
        """
        Inicializa o explorador de séries temporais.
        
        :param df: DataFrame contendo os dados.
        :param date_col: Nome da coluna de data/tempo.
        :param value_col: Nome da coluna com a variável alvo.
        :param freq: Frequência da série (ex: 'D', 'M', 'H'). Se None, tentará inferir.
        """
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(by=self.date_col).set_index(self.date_col)
        
        if freq:
            self.df = self.df.asfreq(freq)
        else:
            try:
                self.df.index.freq = pd.infer_freq(self.df.index)
            except:
                warnings.warn("Não foi possível inferir a frequência automaticamente.")

        self.series = self.df[self.value_col]
        self.results_dict = {}
        self.figures = {}

    def _check_missing_values(self):
        """Análise de completude dos dados."""
        total = len(self.series)
        missing = self.series.isnull().sum()
        return {
            "total_observations": total,
            "missing_values": missing,
            "missing_percentage": round((missing / total) * 100, 2)
        }

    def _descriptive_stats(self):
        """Estatísticas descritivas e momentos de ordem superior."""
        clean_series = self.series.dropna()
        
        desc = clean_series.describe().to_dict()
        desc['skewness'] = clean_series.skew()  # Assimetria
        desc['kurtosis'] = clean_series.kurtosis()  # Curtose (caudas pesadas)
        
        # Teste de Normalidade (Jarque-Bera via Scipy - simplificado para Shapiro se N < 5000)
        if len(clean_series) < 5000:
            stat, p_val = stats.shapiro(clean_series)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_val = stats.jarque_bera(clean_series)
            test_name = "Jarque-Bera"
            
        desc['normality_test'] = {
            "test": test_name,
            "statistic": stat,
            "p_value": p_val,
            "is_normal_0.05": p_val > 0.05
        }
        return desc

    def _stationarity_test(self):
        """Teste de Dickey-Fuller Aumentado (ADF) para raiz unitária."""
        clean_series = self.series.dropna()
        
        # ADF Test
        adf_result = adfuller(clean_series, autolag='AIC')
        
        return {
            "ADF_Statistic": adf_result[0],
            "p_value": adf_result[1],
            "used_lag": adf_result[2],
            "is_stationary_0.05": adf_result[1] < 0.05,
            "interpretation": "Série Estacionária" if adf_result[1] < 0.05 else "Série Não Estacionária (Possui Tendência ou Raiz Unitária)"
        }

    def _plot_diagnostics(self):
        """Gera visualizações essenciais para análise técnica e de negócio."""
        clean_series = self.series.dropna()
        figs = {}

        # 1. Plot da Série Temporal (Histórico)
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(clean_series.index, clean_series.values, label='Observado', color='#1f77b4')
        ax1.set_title(f'Evolução Temporal: {self.value_col}', fontsize=14)
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Valor')
        ax1.legend()
        figs['history_plot'] = fig1

        # 2. Distribuição (Histograma e Densidade)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(clean_series, kde=True, ax=ax2, color='#2ca02c')
        ax2.set_title('Distribuição dos Dados (Análise de Normalidade)', fontsize=14)
        figs['distribution_plot'] = fig2

        # 3. Boxplot por Ano/Mês (Sazonalidade Visual)
        # Criando colunas auxiliares temporárias
        temp_df = pd.DataFrame({'val': clean_series})
        temp_df['year'] = temp_df.index.year
        temp_df['month'] = temp_df.index.month
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=temp_df, x='month', y='val', ax=ax3, palette="viridis")
        ax3.set_title('Sazonalidade: Distribuição de Valores por Mês', fontsize=14)
        figs['seasonality_boxplot'] = fig3

        # 4. Autocorrelação (ACF e PACF)
        fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(clean_series, ax=ax4a, lags=40, title="Autocorrelação (ACF) - Memória da Série")
        plot_pacf(clean_series, ax=ax4b, lags=40, title="Autocorrelação Parcial (PACF) - Dependência Direta")
        plt.tight_layout()
        figs['autocorrelation_plot'] = fig4
        
        # 5. Decomposição Clássica (Se possível)
        try:
            decomp = seasonal_decompose(clean_series.interpolate(method='linear'), model='additive', period=None)
            fig5 = decomp.plot()
            fig5.set_size_inches(12, 10)
            fig5.suptitle('Decomposição (Tendência + Sazonalidade + Resíduo)', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            figs['decomposition_plot'] = fig5
        except Exception as e:
            warnings.warn(f"Não foi possível gerar decomposição: {e}")
            figs['decomposition_plot'] = None

        return figs

    def run_analysis(self):
        """
        Executa a pipeline completa de análise exploratória.
        Retorna um dicionário com dados e visualizações.
        """
        print("Iniciando Análise de Série Temporal...")
        
        # Estrutura e Qualidade
        self.results_dict['data_quality'] = self._check_missing_values()
        
        # Estatísticas
        self.results_dict['statistics'] = self._descriptive_stats()
        
        # Testes de Hipótese (Estacionariedade)
        self.results_dict['stationarity'] = self._stationarity_test()
        
        # Geração de Gráficos
        self.figures = self._plot_diagnostics()
        
        # Empacotamento final
        final_report = {
            "report_summary": self.results_dict,
            "visualizations": self.figures
        }
        
        print("Análise concluída com sucesso.")
        return final_report