import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

sns.set(style="whitegrid")

class NoiseQualityEvaluator:
    def __init__(self, df: pd.DataFrame = None, residuals: pd.Series = None):
        """
        :param df: Opcional se passar residuals diretamente.
        :param residuals: Série temporal dos erros do modelo (y_true - y_pred).
        """
        self.residuals = residuals
        
        # Se não passou resíduos, tenta extrair do DF (método legado/fallback)
        if self.residuals is None and df is not None:
             # Lógica simples de diferenciação como fallback
             self.residuals = df.iloc[:, 0].diff().dropna()

    def set_model_residuals(self, residuals: pd.Series):
        """Conecta com o Forecaster."""
        self.residuals = residuals.dropna()

    def _check_white_noise_properties(self):
        res = self.residuals
        if len(res) < 3 or res.nunique() <= 1:
            return {"error": "Resíduos insuficientes ou constantes."}

        # 1. Normalidade
        try:
            stat, p_val_norm = stats.shapiro(res) if len(res) < 5000 else stats.jarque_bera(res)
        except:
            p_val_norm = 0

        # 2. Autocorrelação (Ljung-Box)
        try:
            # Lags dinâmicos
            lags = min(10, len(res)//2 - 1)
            if lags > 0:
                lb_test = acorr_ljungbox(res, lags=[lags], return_df=True)
                p_val_lb = lb_test['lb_pvalue'].iloc[0]
            else:
                p_val_lb = 0
        except:
            p_val_lb = 0
        
        # 3. Estacionariedade
        try:
            adf_result = adfuller(res)
            p_val_adf = adf_result[1]
        except:
            p_val_adf = 1.0 # Assumir não estacionário se falhar

        return {
            "normality_p_value": p_val_norm,
            "is_normal": p_val_norm > 0.05,
            "autocorr_p_value": p_val_lb,
            "is_white_noise": p_val_lb > 0.05, # H0: é aleatório. > 0.05 não rejeita H0.
            "stationarity_p_value": p_val_adf,
            "is_stationary": p_val_adf < 0.05,
            "mean": np.mean(res),
            "std": np.std(res)
        }

    def plot_diagnostics(self):
        if self.residuals is None or len(self.residuals) == 0:
            return
            
        res = self.residuals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cronologia
        axes[0,0].plot(res)
        axes[0,0].set_title("Resíduos no Tempo")
        axes[0,0].axhline(0, color='red', linestyle='--')
        
        # Histograma
        sns.histplot(res, kde=True, ax=axes[0,1])
        axes[0,1].set_title("Distribuição dos Erros")
        
        # ACF
        lags = min(20, len(res)//2 - 1)
        if lags > 0:
            acf_vals = acf(res, nlags=lags)
            axes[1,0].bar(range(len(acf_vals)), acf_vals)
            axes[1,0].set_title("Autocorrelação dos Resíduos")
            axes[1,0].axhline(1.96/np.sqrt(len(res)), color='r', ls='--')
            axes[1,0].axhline(-1.96/np.sqrt(len(res)), color='r', ls='--')
        
        # QQ Plot
        stats.probplot(res, dist="norm", plot=axes[1,1])
        axes[1,1].set_title("Q-Q Plot")
        
        plt.tight_layout()
        plt.show()

    def run(self):
        print("\n--- Diagnóstico de Qualidade do Modelo (Análise de Resíduos) ---")
        metrics = self._check_white_noise_properties()
        
        if "error" in metrics:
            print(metrics["error"])
            return
            
        print(f"1. Normalidade dos Erros (p={metrics['normality_p_value']:.4f}): {'SIM' if metrics['is_normal'] else 'NÃO'}")
        print(f"2. Ausência de Padrões/Autocorrelação (p={metrics['autocorr_p_value']:.4f}): {'SIM (Bom Modelo)' if metrics['is_white_noise'] else 'NÃO (Modelo deixou padrão)'}")
        print(f"3. Média dos Erros: {metrics['mean']:.4f} (Ideal: 0)")
        
        self.plot_diagnostics()