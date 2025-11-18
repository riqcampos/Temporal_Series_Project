import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class TimeSeriesExplorer:
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str, freq: str = None):
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
                pass

        # Preenchimento inteligente para não quebrar análises
        self.series = self.df[self.value_col].ffill().bfill()
        self.results_dict = {}
        self.figures = {}

    def _check_missing_values(self):
        total = len(self.series)
        missing = self.series.isnull().sum()
        return {
            "total_observations": total,
            "missing_values": missing,
            "missing_percentage": round((missing / total) * 100, 2)
        }

    def _descriptive_stats(self):
        desc = self.series.describe().to_dict()
        desc['skewness'] = self.series.skew()
        desc['kurtosis'] = self.series.kurtosis()
        
        if len(self.series) < 3: # Proteção para séries muito curtas
            return desc

        if len(self.series) < 5000:
            stat, p_val = stats.shapiro(self.series)
            test_name = "Shapiro-Wilk"
        else:
            stat, p_val = stats.jarque_bera(self.series)
            test_name = "Jarque-Bera"
            
        desc['normality_test'] = {
            "test": test_name,
            "statistic": stat,
            "p_value": p_val,
            "is_normal_0.05": p_val > 0.05
        }
        return desc

    def _stationarity_test(self):
        # ADF falha se a série for constante ou muito curta
        if self.series.nunique() <= 1 or len(self.series) < 10:
            return {"error": "Dados insuficientes ou constantes para teste ADF"}

        try:
            adf_result = adfuller(self.series, autolag='AIC')
            return {
                "ADF_Statistic": adf_result[0],
                "p_value": adf_result[1],
                "is_stationary_0.05": adf_result[1] < 0.05
            }
        except Exception as e:
            return {"error": str(e)}

    def _plot_diagnostics(self):
        figs = {}
        
        # 1. Histórico
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(self.series.index, self.series.values, label='Observado', color='#1f77b4')
        ax1.set_title(f'Evolução Temporal: {self.value_col}')
        figs['history_plot'] = fig1

        # 2. Distribuição
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        try:
            sns.histplot(self.series, kde=True, ax=ax2, color='#2ca02c')
            ax2.set_title('Distribuição dos Dados')
            figs['distribution_plot'] = fig2
        except:
            pass

        # 3. Sazonalidade (Boxplot)
        if len(self.series) > 12:
            temp_df = pd.DataFrame({'val': self.series})
            temp_df['month'] = temp_df.index.month
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            # CORREÇÃO DO SEABORN (hue)
            sns.boxplot(data=temp_df, x='month', y='val', hue='month', ax=ax3, palette="viridis", legend=False)
            ax3.set_title('Sazonalidade Mensal')
            figs['seasonality_boxplot'] = fig3

        # 4. ACF e PACF Dinâmicos (CORREÇÃO MATEMÁTICA DO LAG)
        n_samples = len(self.series)
        # Regra de ouro: Lags não podem ser > 50% da amostra. Usamos min(40, N/2 - 1)
        dynamic_lags = min(40, int(n_samples / 2) - 1)
        
        if dynamic_lags > 1:
            fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 10))
            plot_acf(self.series, ax=ax4a, lags=dynamic_lags, title=f"Autocorrelação (Lags={dynamic_lags})")
            plot_pacf(self.series, ax=ax4b, lags=dynamic_lags, title=f"Autocorrelação Parcial (Lags={dynamic_lags})")
            plt.tight_layout()
            figs['autocorrelation_plot'] = fig4
        
        return figs

    def run_analysis(self):
        print(f"--- Explorando série: {self.value_col} ({len(self.series)} amostras) ---")
        self.results_dict['data_quality'] = self._check_missing_values()
        self.results_dict['statistics'] = self._descriptive_stats()
        self.results_dict['stationarity'] = self._stationarity_test()
        self.figures = self._plot_diagnostics()
        
        return {"summary": self.results_dict, "plots": self.figures}