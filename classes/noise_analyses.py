import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

# Configuração de estilo profissional
sns.set(style="whitegrid")

class NoiseQualityEvaluator:
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str, freq: str = None):
        """
        Especialista em análise de componentes estocásticos e resíduos.
        """
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        
        # Tratamento inicial
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(by=self.date_col).set_index(self.date_col)
        
        # Inferência de frequência se não fornecida
        if freq:
            self.df = self.df.asfreq(freq)
        else:
            try:
                inferred_freq = pd.infer_freq(self.df.index)
                if inferred_freq:
                    self.df = self.df.asfreq(inferred_freq)
            except:
                pass # Segue sem frequência definida se falhar
        
        # Preenchimento linear para permitir decomposição
        self.df[self.value_col] = self.df[self.value_col].interpolate(method='linear')
        
        self.residuals = None
        self.metrics = {}

    def _extract_noise(self):
        """
        Separa o sinal do ruído usando decomposição clássica.
        Se a série tiver tendência/sazonalidade, o ruído é o componente residual.
        """
        try:
            # Tenta decomposição aditiva (mais comum para análise geral)
            decomposition = seasonal_decompose(self.df[self.value_col], model='additive', period=None)
            self.residuals = decomposition.resid.dropna()
            self.trend = decomposition.trend
            self.seasonal = decomposition.seasonal
        except Exception as e:
            warnings.warn(f"Decomposição falhou ({str(e)}). Assumindo diferenciação simples para extração de ruído.")
            self.residuals = self.df[self.value_col].diff().dropna()

    def _check_white_noise_properties(self):
        """Verifica as propriedades matemáticas fundamentais do ruído."""
        res = self.residuals
        
        # 1. Teste de Normalidade (Shapiro-Wilk)
        # H0: Os dados seguem uma distribuição normal
        shapiro_stat, shapiro_p = stats.shapiro(res) if len(res) < 5000 else stats.jarque_bera(res)
        
        # 2. Teste de Autocorrelação (Ljung-Box)
        # H0: Os dados são independentes (não há correlação serial) -> Queremos aceitar H0 para ser Ruído Branco
        lb_test = acorr_ljungbox(res, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[0]
        
        # 3. Estacionariedade (ADF)
        # O ruído DEVE ser estacionário.
        adf_result = adfuller(res)
        
        # 4. Estatísticas Básicas
        mean_val = np.mean(res)
        std_val = np.std(res)
        
        return {
            "normality": {
                "test": "Shapiro-Wilk" if len(res) < 5000 else "Jarque-Bera",
                "p_value": shapiro_p,
                "is_gaussian": shapiro_p > 0.05 # Se > 0.05, é Normal
            },
            "independence": {
                "test": "Ljung-Box",
                "p_value": lb_pvalue,
                "is_independent": lb_pvalue > 0.05 # Se > 0.05, é independente (Bom ruído)
            },
            "stationarity": {
                "p_value": adf_result[1],
                "is_stationary": adf_result[1] < 0.05
            },
            "moments": {
                "mean": mean_val, # Deve ser próximo de 0
                "std": std_val,
                "is_zero_mean": np.isclose(mean_val, 0, atol=0.1 * std_val)
            }
        }

    def _calculate_snr(self):
        """Calcula a Relação Sinal-Ruído (Signal-to-Noise Ratio)."""
        # Sinal = Dados Originais - Ruído (ou seja, Tendência + Sazonalidade)
        signal_power = np.var(self.df[self.value_col] - self.residuals)
        noise_power = np.var(self.residuals)
        
        if noise_power == 0:
            return np.inf
            
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def _plot_noise_diagnostics(self):
        """Gera um painel completo de visualização do ruído."""
        res = self.residuals
        
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)

        # 1. Plot da Série de Resíduos (Visualização no Tempo)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(res.index, res.values, color='#d62728', alpha=0.8, lw=1)
        ax1.axhline(0, color='black', linestyle='--', lw=1)
        ax1.set_title('1. Cronologia do Ruído (Resíduos)', fontsize=14)
        ax1.set_ylabel('Amplitude')

        # 2. Histograma vs Curva Normal (Verificação de Gaussianidade)
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(res, kde=True, stat="density", color="gray", ax=ax2)
        
        # Plotando a curva normal teórica por cima
        xmin, xmax = ax2.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(res), np.std(res))
        ax2.plot(x, p, 'k', linewidth=2, label='Normal Teórica')
        ax2.set_title('2. Distribuição do Ruído vs Normal', fontsize=12)
        ax2.legend()

        # 3. Q-Q Plot (Teste Visual de Normalidade)
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(res, dist="norm", plot=ax3)
        ax3.set_title('3. Q-Q Plot (Normalidade)', fontsize=12)

        # 4. Autocorrelação (ACF) - Verificação de memória residual
        ax4 = fig.add_subplot(gs[2, 0])
        # Calculando ACF manualmente para plotar bonito
        acf_vals = acf(res, nlags=20)
        ax4.bar(range(len(acf_vals)), acf_vals, color='#1f77b4')
        ax4.axhline(0, color='black', lw=0.5)
        ax4.axhline(1.96/np.sqrt(len(res)), color='red', linestyle='--', alpha=0.5) # Intervalo de confiança
        ax4.axhline(-1.96/np.sqrt(len(res)), color='red', linestyle='--', alpha=0.5)
        ax4.set_title('4. Autocorrelação (ACF) do Ruído', fontsize=12)
        ax4.set_xlabel('Lags')

        # 5. Volatilidade Móvel (Verificação de Homocedasticidade)
        ax5 = fig.add_subplot(gs[2, 1])
        rolling_std = res.rolling(window=int(len(res)*0.05)).std() # Janela de 5% dos dados
        ax5.plot(rolling_std.index, rolling_std.values, color='purple')
        ax5.set_title('5. Volatilidade Móvel (Busca de Clusters de Variância)', fontsize=12)
        
        plt.tight_layout()
        return fig

    def run_full_evaluation(self):
        """Executa a pipeline completa de avaliação."""
        print("Extraindo componentes e isolando ruído...")
        self._extract_noise()
        
        if len(self.residuals) == 0:
            return "Erro: Não foi possível isolar resíduos suficientes."
            
        print("Calculando estatísticas de ruído branco...")
        stats_metrics = self._check_white_noise_properties()
        
        print("Calculando Relação Sinal-Ruído (SNR)...")
        snr = self._calculate_snr()
        
        self.metrics = {
            "white_noise_metrics": stats_metrics,
            "signal_to_noise_ratio_db": round(snr, 2),
            "interpretation": []
        }
        
        # Interpretação Automática para o Usuário
        interp = self.metrics['interpretation']
        if stats_metrics['normality']['is_gaussian']:
            interp.append("SUCCESS: O ruído segue uma distribuição Normal (Gaussiana).")
        else:
            interp.append("WARNING: O ruído NÃO é Normal (pode haver outliers ou assimetria).")
            
        if stats_metrics['independence']['is_independent']:
            interp.append("SUCCESS: O ruído é aleatório (sem correlação serial). Informação bem extraída.")
        else:
            interp.append("CRITICAL: O ruído tem padrões ocultos (autocorrelação). O modelo anterior deixou sobrar sinal.")
            
        if not stats_metrics['moments']['is_zero_mean']:
            interp.append("WARNING: O ruído tem viés (média não é zero).")

        # Gera os gráficos
        print("Gerando diagnósticos visuais...")
        plot_fig = self._plot_noise_diagnostics()
        
        return {
            "metrics": self.metrics,
            "plot": plot_fig
        }