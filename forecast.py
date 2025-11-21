import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AnalisadorSerieTemporalCompleto:
    """
    Classe completa para análise de séries temporais com validação rigorosa,
    múltiplas visualizações e relatórios para audiências técnicas e não-técnicas.
    """
    
    def __init__(self, df, data_col, valor_col, frequencia='MS', test_size=0.2):
        """
        Inicializa o analisador com validações robustas.
        
        Args:
            df: DataFrame bruto
            data_col: Nome da coluna de data
            valor_col: Nome da coluna de valor (target)
            frequencia: Frequência da série ('MS', 'D', 'W', 'Q', 'Y', etc)
            test_size: Proporção dos dados para teste (validação out-of-sample)
        """
        self.df = df.copy()
        self.df[data_col] = pd.to_datetime(self.df[data_col])
        self.df = self.df.set_index(data_col).sort_index()
        self.nome_valor = valor_col
        self.freq = frequencia
        self.test_size = test_size
        
        # Validações iniciais
        self._validar_dados()
        
        # Tratamento de dados faltantes
        self._tratar_dados_faltantes()
        
        # Split treino-teste temporal
        self._split_treino_teste()
        
        # Containers para resultados
        self.modelos = {}
        self.previsoes = {}
        self.metricas = {}
        self.diagnosticos = {}
        
        print(f"{'='*80}")
        print(f"SÉRIE TEMPORAL: {valor_col}")
        print(f"{'='*80}")
        print(f"Período: {self.df.index.min()} até {self.df.index.max()}")
        print(f"Frequência: {frequencia}")
        print(f"Total de observações: {len(self.df)}")
        print(f"Treino: {len(self.y_train)} | Teste: {len(self.y_test)}")
        print(f"{'='*80}\n")

    def _validar_dados(self):
        """Validações rigorosas dos dados de entrada."""
        # Verificar valores negativos
        if (self.df[self.nome_valor] < 0).any():
            print("AVISO: Série contém valores negativos")
        
        # Verificar valores zero
        zeros = (self.df[self.nome_valor] == 0).sum()
        if zeros > 0:
            print(f"AVISO: {zeros} valores zero encontrados ({zeros/len(self.df)*100:.1f}%)")
        
        # Verificar duplicatas no índice
        if self.df.index.duplicated().any():
            print("AVISO: Índice contém datas duplicadas. Removendo duplicatas...")
            self.df = self.df[~self.df.index.duplicated(keep='first')]
        
        # Definir frequência
        self.df = self.df.asfreq(self.freq)

    def _tratar_dados_faltantes(self):
        """Tratamento robusto de dados faltantes com relatório detalhado."""
        nulos_antes = self.df[self.nome_valor].isna().sum()
        
        if nulos_antes > 0:
            print(f"\n📊 DADOS FALTANTES DETECTADOS")
            print(f"Quantidade: {nulos_antes} ({nulos_antes/len(self.df)*100:.2f}%)")
            print(f"Método de interpolação: linear")
            
            self.df[self.nome_valor] = self.df[self.nome_valor].interpolate(
                method='linear', limit_direction='both'
            )
            
            # Se ainda houver nulos, usar forward/backward fill
            if self.df[self.nome_valor].isna().any():
                self.df[self.nome_valor].fillna(method='ffill', inplace=True)
                self.df[self.nome_valor].fillna(method='bfill', inplace=True)
            
            print(f"✓ Interpolação concluída\n")

    def _split_treino_teste(self):
        """Split temporal preservando a ordem cronológica."""
        n = len(self.df)
        split_idx = int(n * (1 - self.test_size))
        
        self.y_train = self.df[self.nome_valor].iloc[:split_idx]
        self.y_test = self.df[self.nome_valor].iloc[split_idx:]
        
        self.train_df = self.df.iloc[:split_idx]
        self.test_df = self.df.iloc[split_idx:]

    # ==================== ANÁLISE EXPLORATÓRIA ====================
    
    def analise_exploratoria_completa(self):
        """Análise exploratória abrangente com múltiplas visualizações."""
        print(f"\n{'='*80}")
        print("1. ANÁLISE EXPLORATÓRIA DE DADOS")
        print(f"{'='*80}\n")
        
        self._estatisticas_descritivas()
        self._plotar_serie_original()
        self._analise_distribuicao()
        self._detectar_outliers()
        self._analise_tendencia_sazonalidade()

    def _estatisticas_descritivas(self):
        """Estatísticas descritivas detalhadas."""
        y = self.df[self.nome_valor]
        
        print("ESTATÍSTICAS DESCRITIVAS")
        print("-" * 50)
        stats_dict = {
            'Média': y.mean(),
            'Mediana': y.median(),
            'Desvio Padrão': y.std(),
            'Coef. Variação': (y.std() / y.mean()) * 100,
            'Mínimo': y.min(),
            'Máximo': y.max(),
            'Amplitude': y.max() - y.min(),
            'Q1 (25%)': y.quantile(0.25),
            'Q3 (75%)': y.quantile(0.75),
            'IQR': y.quantile(0.75) - y.quantile(0.25),
            'Assimetria': y.skew(),
            'Curtose': y.kurtosis()
        }
        
        for k, v in stats_dict.items():
            print(f"{k:.<40} {v:>15.2f}")
        
        # Análise de variação
        variacao_abs = y.iloc[-1] - y.iloc[0]
        variacao_pct = ((y.iloc[-1] - y.iloc[0]) / y.iloc[0]) * 100
        print(f"\n{'Variação Total (Absoluta)':.<40} {variacao_abs:>15.2f}")
        print(f"{'Variação Total (%)':.<40} {variacao_pct:>15.2f}%")
        print("-" * 50 + "\n")

    def _plotar_serie_original(self):
        """Visualização completa da série original."""
        y = self.df[self.nome_valor]
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # 1. Série temporal completa
        axes[0].plot(y.index, y, linewidth=2, label='Série Original', color='#2E86AB')
        axes[0].axvline(self.y_train.index[-1], color='red', linestyle='--', 
                       linewidth=2, label='Split Treino/Teste')
        axes[0].fill_between(self.y_train.index, y.min(), y.max(), 
                            alpha=0.1, color='green', label='Treino')
        axes[0].fill_between(self.y_test.index, y.min(), y.max(), 
                            alpha=0.1, color='orange', label='Teste')
        axes[0].set_title(f'Série Temporal: {self.nome_valor}', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Valor', fontsize=12)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Média móvel e tendência
        ma_12 = y.rolling(window=12).mean()
        ma_24 = y.rolling(window=24).mean() if len(y) > 24 else None
        
        axes[1].plot(y.index, y, alpha=0.4, label='Original', color='gray')
        axes[1].plot(ma_12.index, ma_12, linewidth=2.5, label='Média Móvel 12', color='#A23B72')
        if ma_24 is not None:
            axes[1].plot(ma_24.index, ma_24, linewidth=2.5, label='Média Móvel 24', color='#F18F01')
        axes[1].set_title('Tendência com Médias Móveis', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Valor', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Variação percentual ano a ano
        if len(y) >= 12:
            variacao_yoy = y.pct_change(periods=12) * 100
            axes[2].plot(variacao_yoy.index, variacao_yoy, linewidth=2, 
                        color='#C73E1D', marker='o', markersize=4)
            axes[2].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[2].fill_between(variacao_yoy.index, 0, variacao_yoy, 
                               where=(variacao_yoy >= 0), alpha=0.3, color='green', 
                               label='Crescimento')
            axes[2].fill_between(variacao_yoy.index, 0, variacao_yoy, 
                               where=(variacao_yoy < 0), alpha=0.3, color='red', 
                               label='Queda')
            axes[2].set_title('Variação Percentual Ano sobre Ano', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Variação (%)', fontsize=12)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _analise_distribuicao(self):
        """Análise da distribuição dos dados."""
        y = self.df[self.nome_valor]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Histograma
        axes[0, 0].hist(y, bins=30, density=True, alpha=0.7, color='#4ECDC4', edgecolor='black')
        axes[0, 0].axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {y.mean():.2f}')
        axes[0, 0].axvline(y.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {y.median():.2f}')
        
        # Ajustar distribuição normal
        mu, sigma = y.mean(), y.std()
        x = np.linspace(y.min(), y.max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal Teórica')
        axes[0, 0].set_title('Distribuição dos Valores', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Valor')
        axes[0, 0].set_ylabel('Densidade')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        stats.probplot(y, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normalidade)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Boxplot
        axes[1, 0].boxplot(y, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='#FFE66D', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
        axes[1, 0].set_title('Boxplot - Detecção de Outliers', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Valor')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Boxplot por período (mensal/trimestral)
        if self.freq == 'MS':
            monthly_data = [y[y.index.month == m].values for m in range(1, 13)]
            axes[1, 1].boxplot(monthly_data, labels=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                                                     'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
                              patch_artist=True)
            axes[1, 1].set_title('Variação Sazonal por Mês', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Mês')
        else:
            yearly_data = [y[y.index.year == yr].values for yr in y.index.year.unique()]
            axes[1, 1].boxplot(yearly_data, labels=y.index.year.unique())
            axes[1, 1].set_title('Variação por Ano', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Ano')
        
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Teste de normalidade
        jb_stat, jb_pval, skew, kurtosis = jarque_bera(y.dropna())
        print(f"TESTE DE NORMALIDADE (Jarque-Bera)")
        print(f"Estatística: {jb_stat:.4f}")
        print(f"P-valor: {jb_pval:.4f}")
        print(f"Conclusão: {'Distribuição NORMAL' if jb_pval > 0.05 else 'Distribuição NÃO-NORMAL'} (α=0.05)\n")

    def _detectar_outliers(self):
        """Detecção e visualização de outliers."""
        y = self.df[self.nome_valor]
        
        # Método IQR
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = y[(y < lower_bound) | (y > upper_bound)]
        
        # Método Z-Score
        z_scores = np.abs(stats.zscore(y))
        outliers_zscore = y[z_scores > 3]
        
        print(f"DETECÇÃO DE OUTLIERS")
        print(f"Método IQR: {len(outliers_iqr)} outliers ({len(outliers_iqr)/len(y)*100:.2f}%)")
        print(f"Método Z-Score (>3): {len(outliers_zscore)} outliers ({len(outliers_zscore)/len(y)*100:.2f}%)")
        
        if len(outliers_iqr) > 0:
            print(f"\nOutliers detectados (IQR):")
            for idx, val in outliers_iqr.items():
                print(f"  {idx.strftime('%Y-%m-%d')}: {val:.2f}")
        print()
        
        # Visualização
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(y.index, y, linewidth=2, label='Série Original', color='#2E86AB')
        ax.scatter(outliers_iqr.index, outliers_iqr, color='red', s=100, 
                  marker='o', label=f'Outliers IQR ({len(outliers_iqr)})', zorder=5)
        ax.axhline(upper_bound, color='red', linestyle='--', alpha=0.5, label='Limites IQR')
        ax.axhline(lower_bound, color='red', linestyle='--', alpha=0.5)
        ax.fill_between(y.index, lower_bound, upper_bound, alpha=0.1, color='green')
        ax.set_title('Detecção de Outliers (Método IQR)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _analise_tendencia_sazonalidade(self):
        """Decomposição sazonal completa (clássica e STL)."""
        y = self.y_train
        
        print(f"\n{'='*80}")
        print("2. DECOMPOSIÇÃO SAZONAL")
        print(f"{'='*80}\n")
        
        # Decomposição clássica
        try:
            decomp_add = seasonal_decompose(y, model='additive', extrapolate_trend='freq')
            decomp_mult = seasonal_decompose(y, model='multiplicative', extrapolate_trend='freq')
            
            fig, axes = plt.subplots(4, 2, figsize=(18, 14))
            
            # Aditiva
            axes[0, 0].plot(y, label='Original', color='black')
            axes[0, 0].set_ylabel('Original')
            axes[0, 0].set_title('Decomposição ADITIVA', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[1, 0].plot(decomp_add.trend, label='Tendência', color='#E63946')
            axes[1, 0].set_ylabel('Tendência')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[2, 0].plot(decomp_add.seasonal, label='Sazonalidade', color='#06FFA5')
            axes[2, 0].set_ylabel('Sazonalidade')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            axes[3, 0].plot(decomp_add.resid, label='Resíduo', color='#F4A261')
            axes[3, 0].axhline(0, color='black', linestyle='--')
            axes[3, 0].set_ylabel('Resíduo')
            axes[3, 0].set_xlabel('Data')
            axes[3, 0].legend()
            axes[3, 0].grid(True, alpha=0.3)
            
            # Multiplicativa
            axes[0, 1].plot(y, label='Original', color='black')
            axes[0, 1].set_ylabel('Original')
            axes[0, 1].set_title('Decomposição MULTIPLICATIVA', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 1].plot(decomp_mult.trend, label='Tendência', color='#E63946')
            axes[1, 1].set_ylabel('Tendência')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[2, 1].plot(decomp_mult.seasonal, label='Sazonalidade', color='#06FFA5')
            axes[2, 1].set_ylabel('Sazonalidade')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            axes[3, 1].plot(decomp_mult.resid, label='Resíduo', color='#F4A261')
            axes[3, 1].axhline(1, color='black', linestyle='--')
            axes[3, 1].set_ylabel('Resíduo')
            axes[3, 1].set_xlabel('Data')
            axes[3, 1].legend()
            axes[3, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Calcular força da sazonalidade
            strength_seasonal_add = 1 - (decomp_add.resid.var() / (decomp_add.resid + decomp_add.seasonal).var())
            strength_trend_add = 1 - (decomp_add.resid.var() / (decomp_add.resid + decomp_add.trend).var())
            
            print(f"FORÇA DOS COMPONENTES (Aditivo)")
            print(f"Força da Sazonalidade: {strength_seasonal_add:.4f}")
            print(f"Força da Tendência: {strength_trend_add:.4f}\n")
            
        except Exception as e:
            print(f"Não foi possível realizar decomposição clássica: {e}\n")
        
        # Decomposição STL (mais robusta)
        try:
            if len(y) >= 24:
                stl = STL(y, seasonal=13)
                result_stl = stl.fit()
                
                fig, axes = plt.subplots(4, 1, figsize=(16, 12))
                
                axes[0].plot(y, label='Original', color='black')
                axes[0].set_ylabel('Original')
                axes[0].set_title('Decomposição STL (Seasonal-Trend decomposition using Loess)', 
                                fontweight='bold', fontsize=14)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(result_stl.trend, label='Tendência', color='#E63946', linewidth=2)
                axes[1].set_ylabel('Tendência')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(result_stl.seasonal, label='Sazonalidade', color='#06FFA5', linewidth=2)
                axes[2].set_ylabel('Sazonalidade')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                axes[3].plot(result_stl.resid, label='Resíduo', color='#F4A261', linewidth=2)
                axes[3].axhline(0, color='black', linestyle='--')
                axes[3].set_ylabel('Resíduo')
                axes[3].set_xlabel('Data')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Não foi possível realizar decomposição STL: {e}\n")

    # ==================== TESTES DE ESTACIONARIEDADE ====================
    
    def testes_estacionariedade(self):
        """Testes completos de estacionariedade."""
        print(f"\n{'='*80}")
        print("3. TESTES DE ESTACIONARIEDADE")
        print(f"{'='*80}\n")
        
        y = self.y_train.dropna()
        
        # 1. Teste ADF (Augmented Dickey-Fuller)
        print("TESTE ADF (Augmented Dickey-Fuller)")
        print("-" * 50)
        adf_result = adfuller(y, autolag='AIC')
        print(f"Estatística ADF: {adf_result[0]:.6f}")
        print(f"P-valor: {adf_result[1]:.6f}")
        print(f"Lags utilizados: {adf_result[2]}")
        print(f"Número de observações: {adf_result[3]}")
        print("Valores críticos:")
        for key, value in adf_result[4].items():
            print(f"  {key}: {value:.4f}")
        
        if adf_result[1] < 0.05:
            print("Conclusão: Série é ESTACIONÁRIA (rejeita H0: raiz unitária)")
        else:
            print("Conclusão: Série é NÃO-ESTACIONÁRIA (não rejeita H0)")
        
        # 2. Teste KPSS
        print(f"\nTESTE KPSS (Kwiatkowski-Phillips-Schmidt-Shin)")
        print("-" * 50)
        kpss_result = kpss(y, regression='ct', nlags='auto')
        print(f"Estatística KPSS: {kpss_result[0]:.6f}")
        print(f"P-valor: {kpss_result[1]:.6f}")
        print(f"Lags utilizados: {kpss_result[2]}")
        print("Valores críticos:")
        for key, value in kpss_result[3].items():
            print(f"  {key}: {value:.4f}")
        
        if kpss_result[1] > 0.05:
            print("Conclusão: Série é ESTACIONÁRIA (não rejeita H0: estacionária)")
        else:
            print("Conclusão: Série é NÃO-ESTACIONÁRIA (rejeita H0)")
        
        print(f"\n{'='*50}")
        print("INTERPRETAÇÃO COMBINADA:")
        if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
            print("Série é ESTACIONÁRIA (ambos os testes concordam)")
        elif adf_result[1] >= 0.05 and kpss_result[1] <= 0.05:
            print("Série é NÃO-ESTACIONÁRIA (ambos os testes concordam)")
        else:
            print("Resultados INCONCLUSIVOS (testes discordam)")
        print(f"{'='*50}\n")
        
        # Visualização ACF e PACF
        self._plotar_acf_pacf()

    def _plotar_acf_pacf(self):
        """Plotar ACF e PACF para identificação de modelos."""
        y = self.y_train.dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # ACF e PACF da série original
        plot_acf(y, ax=axes[0, 0], lags=min(40, len(y)//2), alpha=0.05)
        axes[0, 0].set_title('ACF - Série Original', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        plot_pacf(y, ax=axes[0, 1], lags=min(40, len(y)//2), alpha=0.05, method='ywm')
        axes[0, 1].set_title('PACF - Série Original', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF e PACF da primeira diferença
        y_diff = y.diff().dropna()
        plot_acf(y_diff, ax=axes[1, 0], lags=min(40, len(y_diff)//2), alpha=0.05)
        axes[1, 0].set_title('ACF - Primeira Diferença', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        plot_pacf(y_diff, ax=axes[1, 1], lags=min(40, len(y_diff)//2), alpha=0.05, method='ywm')
        axes[1, 1].set_title('PACF - Primeira Diferença', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # ==================== MODELAGEM ====================
    
    def ajustar_todos_modelos(self, horizonte=None):
        """
        Ajusta todos os modelos disponíveis e compara performance.
        
        Args:
            horizonte: Número de períodos futuros para previsão (default: tamanho do teste)
        """
        if horizonte is None:
            horizonte = len(self.y_test)
        
        print(f"\n{'='*80}")
        print("4. AJUSTE E COMPARAÇÃO DE MODELOS")
        print(f"{'='*80}\n")
        
        # Ajustar cada modelo
        self.aplicar_random_forest(horizonte)
        self.aplicar_holt_winters(horizonte)
        self.aplicar_sarima(horizonte)
        self.aplicar_prophet(horizonte)
        
        # Comparação visual e métricas
        self._comparar_modelos()
        self._ranking_modelos()

    def aplicar_random_forest(self, horizonte):
        """Random Forest com features de série temporal (lags e features temporais)."""
        print("RANDOM FOREST REGRESSOR")
        print("-" * 50)
        
        y_train = self.y_train
        y_test = self.y_test
        
        try:
            # Criar features de série temporal
            def criar_features(data, n_lags=12):
                df = pd.DataFrame(index=data.index)
                df['valor'] = data.values
                
                # Lags
                for i in range(1, n_lags + 1):
                    df[f'lag_{i}'] = data.shift(i)
                
                # Rolling statistics
                for window in [3, 6, 12]:
                    df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
                    df[f'rolling_std_{window}'] = data.rolling(window=window).std()
                
                # Features temporais
                df['mes'] = data.index.month
                df['trimestre'] = data.index.quarter
                df['ano'] = data.index.year
                
                # Tendência
                df['tendencia'] = np.arange(len(data))
                
                return df
            
            # Criar features para treino
            df_train = criar_features(y_train)
            df_train = df_train.dropna()
            
            X_train = df_train.drop('valor', axis=1)
            y_train_rf = df_train['valor']
            
            # Treinar modelo
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train, y_train_rf)
            
            # Prever no conjunto de teste
            predicoes_test = []
            ultimo_treino = y_train.copy()
            
            for i in range(len(y_test)):
                # Criar features para próxima previsão
                df_pred = criar_features(ultimo_treino)
                df_pred = df_pred.dropna()
                X_pred = df_pred.iloc[[-1]].drop('valor', axis=1)
                
                # Prever
                pred = rf_model.predict(X_pred)[0]
                predicoes_test.append(pred)
                
                # Atualizar série com valor real para próxima iteração
                novo_index = pd.DatetimeIndex([y_test.index[i]])
                novo_valor = pd.Series([y_test.iloc[i]], index=novo_index)
                ultimo_treino = pd.concat([ultimo_treino, novo_valor])
            
            forecast_series = pd.Series(predicoes_test, index=y_test.index)
            
            # Calcular intervalos de confiança (usando desvio padrão das árvores)
            predicoes_arvores = np.array([tree.predict(X_train) for tree in rf_model.estimators_])
            std_pred = np.std(predicoes_arvores, axis=0).mean()
            
            forecast_lower = forecast_series - 1.96 * std_pred
            forecast_upper = forecast_series + 1.96 * std_pred
            
            # Armazenar
            self.modelos['RandomForest'] = rf_model
            self.previsoes['RandomForest'] = forecast_series
            self.previsoes['RandomForest_lower'] = forecast_lower
            self.previsoes['RandomForest_upper'] = forecast_upper
            self.metricas['RandomForest'] = self._calcular_metricas(y_test, forecast_series)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"Random Forest treinado com sucesso")
            print(f"  Número de árvores: {rf_model.n_estimators}")
            print(f"  Features mais importantes:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            # Diagnóstico de resíduos
            fitted = rf_model.predict(X_train)
            residuos = y_train_rf - fitted
            self._diagnostico_residuos(residuos, fitted, 'Random Forest')
            
            print("Random Forest ajustado com sucesso\n")
            
        except Exception as e:
            print(f"Erro no Random Forest: {e}\n")

    def aplicar_holt_winters(self, horizonte):
        """Suavização Exponencial de Holt-Winters (Triple Exponential Smoothing)."""
        print("HOLT-WINTERS (SUAVIZAÇÃO EXPONENCIAL)")
        print("-" * 50)
        
        y_train = self.y_train
        y_test = self.y_test
        
        try:
            # Detectar período sazonal automaticamente
            seasonal_period = 12 if self.freq == 'MS' else 4 if self.freq == 'Q' else 7
            
            # Testar modelos aditivo e multiplicativo
            modelos_hw = {}
            for trend in ['add', 'mul']:
                for seasonal in ['add', 'mul']:
                    try:
                        model_name = f"HW_{trend}_{seasonal}"
                        model = ExponentialSmoothing(
                            y_train, 
                            trend=trend, 
                            seasonal=seasonal, 
                            seasonal_periods=seasonal_period
                        ).fit(optimized=True)
                        
                        modelos_hw[model_name] = model
                    except:
                        pass
            
            # Selecionar melhor modelo por AIC
            if modelos_hw:
                melhor_nome = min(modelos_hw, key=lambda k: modelos_hw[k].aic)
                modelo_hw = modelos_hw[melhor_nome]
                
                print(f"✓ Melhor configuração: {melhor_nome}")
                print(f"  AIC: {modelo_hw.aic:.2f}")
                print(f"  BIC: {modelo_hw.bic:.2f}")
                
                # Previsões
                fitted = modelo_hw.fittedvalues
                forecast = modelo_hw.forecast(len(y_test))
                forecast_series = pd.Series(forecast, index=y_test.index)
                
                # Intervalos de confiança (aproximação)
                residuos = y_train - fitted
                std_resid = residuos.std()
                forecast_lower = forecast_series - 1.96 * std_resid
                forecast_upper = forecast_series + 1.96 * std_resid
                
                # Armazenar
                self.modelos['HoltWinters'] = modelo_hw
                self.previsoes['HoltWinters'] = forecast_series
                self.previsoes['HoltWinters_lower'] = forecast_lower
                self.previsoes['HoltWinters_upper'] = forecast_upper
                self.metricas['HoltWinters'] = self._calcular_metricas(y_test, forecast_series)
                
                # Diagnóstico de resíduos
                self._diagnostico_residuos(residuos, fitted, 'Holt-Winters')
                
                print("✓ Holt-Winters ajustado com sucesso\n")
            else:
                print("✗ Não foi possível ajustar Holt-Winters\n")
                
        except Exception as e:
            print(f"✗ Erro no Holt-Winters: {e}\n")

    def aplicar_sarima(self, horizonte):
        """SARIMA com seleção automática de parâmetros."""
        print("SARIMA (AUTO-ARIMA)")
        print("-" * 50)
        
        y_train = self.y_train
        y_test = self.y_test
        
        try:
            # Determinar m (período sazonal)
            m = 12 if self.freq == 'MS' else 4 if self.freq == 'Q' else 1
            
            print("Buscando melhor modelo SARIMA (pode levar alguns minutos)...")
            
            modelo_sarima = auto_arima(
                y_train,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                m=m,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                seasonal=True if m > 1 else False,
                d=None, D=None,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_fits=50
            )
            
            print(f"Melhor modelo: ARIMA{modelo_sarima.order} x {modelo_sarima.seasonal_order}[{m}]")
            print(f"  AIC: {modelo_sarima.aic():.2f}")
            print(f"  BIC: {modelo_sarima.bic():.2f}")
            
            # Previsões
            forecast, conf_int = modelo_sarima.predict(n_periods=len(y_test), return_conf_int=True)
            forecast_series = pd.Series(forecast, index=y_test.index)
            forecast_lower = pd.Series(conf_int[:, 0], index=y_test.index)
            forecast_upper = pd.Series(conf_int[:, 1], index=y_test.index)
            
            # Fitted values
            fitted = modelo_sarima.predict_in_sample()
            fitted_series = pd.Series(fitted, index=y_train.index)
            
            # Armazenar
            self.modelos['SARIMA'] = modelo_sarima
            self.previsoes['SARIMA'] = forecast_series
            self.previsoes['SARIMA_lower'] = forecast_lower
            self.previsoes['SARIMA_upper'] = forecast_upper
            self.metricas['SARIMA'] = self._calcular_metricas(y_test, forecast_series)
            
            # Diagnóstico de resíduos
            residuos = modelo_sarima.resid()
            self._diagnostico_residuos(residuos, fitted_series, 'SARIMA')
            
            print("SARIMA ajustado com sucesso\n")
            
        except Exception as e:
            print(f"Erro no SARIMA: {e}\n")

    def aplicar_prophet(self, horizonte):
        """Prophet com detecção automática de tendências e sazonalidade."""
        print("PROPHET")
        print("-" * 50)
        
        try:
            # Preparar dados
            df_train = self.train_df.reset_index()
            df_train.columns = ['ds', 'y']
            
            df_test = self.test_df.reset_index()
            df_test.columns = ['ds', 'y']
            
            # Configurar modelo
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            # Ajustar
            model.fit(df_train)
            
            # Previsões
            future = model.make_future_dataframe(periods=len(df_test), freq=self.freq)
            forecast = model.predict(future)
            
            # Separar treino e teste
            forecast_train = forecast.iloc[:len(df_train)]
            forecast_test = forecast.iloc[len(df_train):]
            
            fitted_series = pd.Series(forecast_train['yhat'].values, index=self.y_train.index)
            forecast_series = pd.Series(forecast_test['yhat'].values, index=self.y_test.index)
            forecast_lower = pd.Series(forecast_test['yhat_lower'].values, index=self.y_test.index)
            forecast_upper = pd.Series(forecast_test['yhat_upper'].values, index=self.y_test.index)
            
            # Armazenar
            self.modelos['Prophet'] = model
            self.previsoes['Prophet'] = forecast_series
            self.previsoes['Prophet_lower'] = forecast_lower
            self.previsoes['Prophet_upper'] = forecast_upper
            self.metricas['Prophet'] = self._calcular_metricas(self.y_test, forecast_series)
            
            # Diagnóstico de resíduos
            residuos = df_train['y'] - forecast_train['yhat']
            self._diagnostico_residuos(residuos, fitted_series, 'Prophet')
            
            # Componentes do Prophet
            self._plotar_componentes_prophet(model, forecast)
            
            print("Prophet ajustado com sucesso\n")
            
        except Exception as e:
            print(f"Erro no Prophet: {e}\n")

    def _calcular_metricas(self, y_true, y_pred):
        """Calcula métricas abrangentes de performance."""
        y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
        y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
        
        # Remover NaNs
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {}
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # SMAPE (Symmetric MAPE)
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'Bias': bias,
            'R²': r2
        }

    def _diagnostico_residuos(self, residuos, fitted, nome_modelo):
        """Diagnóstico completo de resíduos."""
        residuos = pd.Series(residuos).dropna()
        
        if len(residuos) < 10:
            return
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Resíduos ao longo do tempo
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(residuos.index if hasattr(residuos, 'index') else range(len(residuos)), 
                residuos, linewidth=1.5, color='#E63946')
        ax1.axhline(0, color='black', linestyle='--', linewidth=2)
        ax1.fill_between(range(len(residuos)), -1.96*residuos.std(), 1.96*residuos.std(), 
                         alpha=0.2, color='gray', label='±1.96σ')
        ax1.set_title(f'Resíduos ao Longo do Tempo - {nome_modelo}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Resíduo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Histograma
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(residuos, bins=30, density=True, alpha=0.7, color='#4ECDC4', edgecolor='black')
        mu, sigma = residuos.mean(), residuos.std()
        x = np.linspace(residuos.min(), residuos.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        ax2.set_title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Resíduo')
        ax2.set_ylabel('Densidade')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(residuos, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Resíduos vs Fitted
        ax4 = fig.add_subplot(gs[1, 2])
        if hasattr(fitted, '__len__'):
            fitted_values = fitted.values if isinstance(fitted, pd.Series) else fitted
            if len(fitted_values) == len(residuos):
                ax4.scatter(fitted_values, residuos, alpha=0.6, color='#F4A261')
                ax4.axhline(0, color='black', linestyle='--', linewidth=2)
                ax4.set_title('Resíduos vs Valores Ajustados', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Valores Ajustados')
                ax4.set_ylabel('Resíduo')
                ax4.grid(True, alpha=0.3)
        
        # 5. ACF dos resíduos
        ax5 = fig.add_subplot(gs[2, 0])
        plot_acf(residuos, ax=ax5, lags=min(40, len(residuos)//2), alpha=0.05)
        ax5.set_title('ACF dos Resíduos', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. PACF dos resíduos
        ax6 = fig.add_subplot(gs[2, 1])
        plot_pacf(residuos, ax=ax6, lags=min(40, len(residuos)//2), alpha=0.05, method='ywm')
        ax6.set_title('PACF dos Resíduos', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Resíduos quadrados (heterocedasticidade)
        ax7 = fig.add_subplot(gs[2, 2])
        residuos_quad = residuos ** 2
        ax7.plot(residuos_quad.index if hasattr(residuos_quad, 'index') else range(len(residuos_quad)),
                residuos_quad, linewidth=1.5, color='#A23B72')
        ax7.set_title('Resíduos² (Heterocedasticidade)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Tempo')
        ax7.set_ylabel('Resíduo²')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle(f'Diagnóstico Completo de Resíduos - {nome_modelo}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.show()
        
        # Testes estatísticos
        print(f"\nTESTES ESTATÍSTICOS DOS RESÍDUOS - {nome_modelo}")
        print("-" * 60)
        
        # Teste de Normalidade (Jarque-Bera)
        jb_stat, jb_pval, _, _ = jarque_bera(residuos)
        print(f"Teste Jarque-Bera (Normalidade):")
        print(f"  Estatística: {jb_stat:.4f} | P-valor: {jb_pval:.4f}")
        print(f"  {'Resíduos NORMAIS' if jb_pval > 0.05 else '✗ Resíduos NÃO-NORMAIS'}")
        
        # Teste Ljung-Box (Autocorrelação)
        lb_test = acorr_ljungbox(residuos, lags=min(10, len(residuos)//5), return_df=True)
        lb_pval_min = lb_test['lb_pvalue'].min()
        print(f"\nTeste Ljung-Box (Autocorrelação):")
        print(f"  P-valor mínimo: {lb_pval_min:.4f}")
        print(f"  {'SEM autocorrelação (Ruído Branco)' if lb_pval_min > 0.05 else '✗ COM autocorrelação'}")
        
        # Estatísticas descritivas
        print(f"\nEstatísticas dos Resíduos:")
        print(f"  Média: {residuos.mean():.6f} (deveria ser ~0)")
        print(f"  Desvio Padrão: {residuos.std():.4f}")
        print(f"  Assimetria: {residuos.skew():.4f}")
        print(f"  Curtose: {residuos.kurtosis():.4f}")
        print("-" * 60 + "\n")

    def _plotar_componentes_prophet(self, model, forecast):
        """Visualiza componentes do Prophet."""
        fig = model.plot_components(forecast)
        fig.set_size_inches(16, 10)
        plt.suptitle('Componentes do Modelo Prophet', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()

    def _comparar_modelos(self):
        """Comparação visual de todos os modelos."""
        if not self.previsoes:
            print("Nenhum modelo ajustado ainda.")
            return
        
        print(f"\n{'='*80}")
        print("5. COMPARAÇÃO VISUAL DOS MODELOS")
        print(f"{'='*80}\n")
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        
        # Gráfico 1: Série completa com todos os modelos
        ax1 = axes[0]
        ax1.plot(self.y_train.index, self.y_train, label='Treino', 
                color='black', linewidth=2, alpha=0.7)
        ax1.plot(self.y_test.index, self.y_test, label='Teste (Real)', 
                color='black', linewidth=2.5, linestyle='--')
        
        colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E76F51', '#8AB17D']
        for i, (nome, pred) in enumerate(self.previsoes.items()):
            if '_lower' not in nome and '_upper' not in nome:
                ax1.plot(pred.index, pred, label=nome, 
                        linewidth=2, alpha=0.8, color=colors[i % len(colors)])
        
        ax1.axvline(self.y_train.index[-1], color='red', linestyle=':', 
                   linewidth=2, alpha=0.7, label='Split')
        ax1.set_title('Comparação de Todos os Modelos - Série Completa', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Valor')
        ax1.legend(loc='best', ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Zoom no período de teste
        ax2 = axes[1]
        ax2.plot(self.y_test.index, self.y_test, label='Real', 
                color='black', linewidth=3, marker='o', markersize=6)
        
        for i, (nome, pred) in enumerate(self.previsoes.items()):
            if '_lower' not in nome and '_upper' not in nome:
                ax2.plot(pred.index, pred, label=nome, 
                        linewidth=2.5, marker='s', markersize=5, 
                        alpha=0.7, color=colors[i % len(colors)])
                
                # Adicionar intervalos de confiança se disponíveis
                if f'{nome}_lower' in self.previsoes and f'{nome}_upper' in self.previsoes:
                    ax2.fill_between(pred.index, 
                                    self.previsoes[f'{nome}_lower'],
                                    self.previsoes[f'{nome}_upper'],
                                    alpha=0.15, color=colors[i % len(colors)])
        
        ax2.set_title('Comparação no Período de Teste (Out-of-Sample)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Valor')
        ax2.legend(loc='best', ncol=2)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _ranking_modelos(self):
        """Ranking e tabela de métricas de todos os modelos."""
        if not self.metricas:
            print("Nenhuma métrica calculada ainda.")
            return
        
        print(f"\n{'='*80}")
        print("6. RANKING E MÉTRICAS DE PERFORMANCE")
        print(f"{'='*80}\n")
        
        # Criar DataFrame com métricas
        df_metricas = pd.DataFrame(self.metricas).T
        df_metricas = df_metricas.round(4)
        
        # Ranking por cada métrica (menor é melhor, exceto R²)
        df_metricas['Rank_MAE'] = df_metricas['MAE'].rank()
        df_metricas['Rank_RMSE'] = df_metricas['RMSE'].rank()
        df_metricas['Rank_MAPE'] = df_metricas['MAPE'].rank()
        df_metricas['Rank_R²'] = df_metricas['R²'].rank(ascending=False)
        
        # Ranking médio
        df_metricas['Rank_Médio'] = df_metricas[
            ['Rank_MAE', 'Rank_RMSE', 'Rank_MAPE', 'Rank_R²']
        ].mean(axis=1)
        
        df_metricas = df_metricas.sort_values('Rank_Médio')
        
        # Exibir tabela
        print("TABELA DE MÉTRICAS (ordenada por melhor performance)")
        print("=" * 100)
        print(df_metricas[['MAE', 'RMSE', 'MAPE', 'SMAPE', 'R²', 'Rank_Médio']].to_string())
        print("=" * 100)
        
        # Melhor modelo
        melhor_modelo = df_metricas.index[0]
        print(f"\nMELHOR MODELO: {melhor_modelo}")
        print(f"   MAE: {df_metricas.loc[melhor_modelo, 'MAE']:.4f}")
        print(f"   RMSE: {df_metricas.loc[melhor_modelo, 'RMSE']:.4f}")
        print(f"   MAPE: {df_metricas.loc[melhor_modelo, 'MAPE']:.2f}%")
        print(f"   R²: {df_metricas.loc[melhor_modelo, 'R²']:.4f}")
        
        # Visualização de métricas
        self._plotar_metricas_comparacao(df_metricas)

    def _plotar_metricas_comparacao(self, df_metricas):
        """Visualização gráfica das métricas."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        modelos = df_metricas.index
        colors = plt.cm.viridis(np.linspace(0, 1, len(modelos)))
        
        # MAE
        axes[0, 0].barh(modelos, df_metricas['MAE'], color=colors, edgecolor='black')
        axes[0, 0].set_xlabel('MAE (menor é melhor)', fontweight='bold')
        axes[0, 0].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # RMSE
        axes[0, 1].barh(modelos, df_metricas['RMSE'], color=colors, edgecolor='black')
        axes[0, 1].set_xlabel('RMSE (menor é melhor)', fontweight='bold')
        axes[0, 1].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # MAPE
        axes[1, 0].barh(modelos, df_metricas['MAPE'], color=colors, edgecolor='black')
        axes[1, 0].set_xlabel('MAPE % (menor é melhor)', fontweight='bold')
        axes[1, 0].set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # R²
        axes[1, 1].barh(modelos, df_metricas['R²'], color=colors, edgecolor='black')
        axes[1, 1].set_xlabel('R² (maior é melhor)', fontweight='bold')
        axes[1, 1].set_title('Coeficiente de Determinação', fontsize=12, fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Comparação de Métricas de Performance', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ==================== PREVISÃO FUTURA ====================
    
    def prever_futuro(self, n_periodos):
        """
        Gera previsões para períodos futuros usando todos os modelos ajustados.
        
        Args:
            n_periodos: Número de períodos futuros para prever
        
        Returns:
            DataFrame com previsões de todos os modelos
        """
        print(f"\n{'='*80}")
        print(f"PREVISÃO FUTURA - {n_periodos} PERÍODOS À FRENTE")
        print(f"{'='*80}\n")
        
        # Criar índice futuro
        ultimo_periodo = self.df.index[-1]
        if self.freq == 'MS':
            freq_offset = pd.DateOffset(months=1)
        elif self.freq == 'Q':
            freq_offset = pd.DateOffset(months=3)
        elif self.freq == 'D':
            freq_offset = pd.DateOffset(days=1)
        elif self.freq == 'W':
            freq_offset = pd.DateOffset(weeks=1)
        elif self.freq == 'Y':
            freq_offset = pd.DateOffset(years=1)
        else:
            freq_offset = pd.DateOffset(months=1)
        
        datas_futuras = pd.date_range(
            start=ultimo_periodo + freq_offset,
            periods=n_periodos,
            freq=self.freq
        )
        
        previsoes_futuras = pd.DataFrame(index=datas_futuras)
        
        # Random Forest
        if 'RandomForest' in self.modelos:
            print("Gerando previsões com Random Forest...")
            try:
                rf_model = self.modelos['RandomForest']
                serie_completa = self.df[self.nome_valor].copy()
                
                predicoes_rf = []
                for i in range(n_periodos):
                    # Criar features
                    def criar_features_single(data):
                        df = pd.DataFrame(index=[data.index[-1]])
                        
                        # Lags
                        for lag in range(1, 13):
                            df[f'lag_{lag}'] = data.iloc[-lag] if len(data) >= lag else data.iloc[0]
                        
                        # Rolling statistics
                        for window in [3, 6, 12]:
                            if len(data) >= window:
                                df[f'rolling_mean_{window}'] = data.iloc[-window:].mean()
                                df[f'rolling_std_{window}'] = data.iloc[-window:].std()
                            else:
                                df[f'rolling_mean_{window}'] = data.mean()
                                df[f'rolling_std_{window}'] = data.std()
                        
                        # Features temporais
                        next_date = data.index[-1] + freq_offset
                        df['mes'] = next_date.month
                        df['trimestre'] = next_date.quarter
                        df['ano'] = next_date.year
                        df['tendencia'] = len(data) + i
                        
                        return df
                    
                    X_pred = criar_features_single(serie_completa)
                    pred = rf_model.predict(X_pred)[0]
                    predicoes_rf.append(pred)
                    
                    # Adicionar previsão à série
                    nova_data = datas_futuras[i]
                    serie_completa = pd.concat([
                        serie_completa,
                        pd.Series([pred], index=[nova_data])
                    ])
                
                previsoes_futuras['RandomForest'] = predicoes_rf
                print("✓ Random Forest concluído")
                
            except Exception as e:
                print(f"✗ Erro no Random Forest: {e}")
        
        # Holt-Winters
        if 'HoltWinters' in self.modelos:
            print("Gerando previsões com Holt-Winters...")
            try:
                hw_model = self.modelos['HoltWinters']
                previsoes_hw = hw_model.forecast(n_periodos)
                previsoes_futuras['HoltWinters'] = previsoes_hw
                print("✓ Holt-Winters concluído")
            except Exception as e:
                print(f"✗ Erro no Holt-Winters: {e}")
        
        # SARIMA
        if 'SARIMA' in self.modelos:
            print("Gerando previsões com SARIMA...")
            try:
                sarima_model = self.modelos['SARIMA']
                previsoes_sarima = sarima_model.predict(n_periods=n_periodos)
                previsoes_futuras['SARIMA'] = previsoes_sarima
                print("✓ SARIMA concluído")
            except Exception as e:
                print(f"✗ Erro no SARIMA: {e}")
        
        # Prophet
        if 'Prophet' in self.modelos:
            print("Gerando previsões com Prophet...")
            try:
                prophet_model = self.modelos['Prophet']
                future = prophet_model.make_future_dataframe(periods=n_periodos, freq=self.freq)
                forecast = prophet_model.predict(future)
                previsoes_prophet = forecast.tail(n_periodos)['yhat'].values
                previsoes_futuras['Prophet'] = previsoes_prophet
                print("✓ Prophet concluído")
            except Exception as e:
                print(f"✗ Erro no Prophet: {e}")
        
        # Visualização
        self._plotar_previsoes_futuras(previsoes_futuras, n_periodos)
        
        # Estatísticas
        print(f"\n{'='*80}")
        print("RESUMO DAS PREVISÕES FUTURAS")
        print(f"{'='*80}\n")
        print(previsoes_futuras.round(2))
        print("\nEstatísticas por período:")
        print(previsoes_futuras.describe().round(2))
        
        return previsoes_futuras
    
    def _plotar_previsoes_futuras(self, previsoes_futuras, n_periodos):
        """Visualiza as previsões futuras de todos os modelos."""
        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        
        y_historico = self.df[self.nome_valor]
        
        # Gráfico 1: Série completa com previsões
        ax1 = axes[0]
        ax1.plot(y_historico.index, y_historico, label='Histórico', 
                linewidth=2.5, color='black', alpha=0.7)
        
        colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653']
        for i, col in enumerate(previsoes_futuras.columns):
            ax1.plot(previsoes_futuras.index, previsoes_futuras[col], 
                    label=f'{col} (Futuro)', linewidth=2.5, 
                    color=colors[i % len(colors)], marker='o', markersize=6)
        
        ax1.axvline(y_historico.index[-1], color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Início das Previsões')
        ax1.fill_between(previsoes_futuras.index, 
                        previsoes_futuras.min(axis=1), 
                        previsoes_futuras.max(axis=1),
                        alpha=0.2, color='gray', label='Range das Previsões')
        
        ax1.set_title(f'Previsões Futuras - {n_periodos} Períodos', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Valor')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Zoom nas previsões futuras
        ax2 = axes[1]
        
        # Incluir últimos períodos históricos para contexto
        n_contexto = min(12, len(y_historico))
        contexto = y_historico.iloc[-n_contexto:]
        
        ax2.plot(contexto.index, contexto, label='Últimos Valores Históricos', 
                linewidth=2.5, color='black', alpha=0.7, marker='o')
        
        for i, col in enumerate(previsoes_futuras.columns):
            ax2.plot(previsoes_futuras.index, previsoes_futuras[col], 
                    label=col, linewidth=2.5, color=colors[i % len(colors)], 
                    marker='s', markersize=7)
        
        ax2.axvline(y_historico.index[-1], color='red', linestyle='--', 
                   linewidth=2, alpha=0.7)
        
        ax2.set_title('Zoom: Transição Histórico → Futuro', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Valor')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # ==================== RELATÓRIO FINAL ====================
    
    def gerar_relatorio_executivo(self):
        """Gera relatório executivo para audiências não-técnicas."""
        print(f"\n{'='*80}")
        print("7. RELATÓRIO EXECUTIVO")
        print(f"{'='*80}\n")
        
        y = self.df[self.nome_valor]
        
        print("RESUMO DA ANÁLISE")
        print("-" * 80)
        print(f"Variável Analisada: {self.nome_valor}")
        print(f"Período: {y.index.min().strftime('%Y-%m-%d')} até {y.index.max().strftime('%Y-%m-%d')}")
        print(f"Total de Observações: {len(y)}")
        print(f"Frequência: {self.freq}")
        
        # Tendência geral
        if y.iloc[-1] > y.iloc[0]:
            tendencia = f"CRESCIMENTO de {((y.iloc[-1]/y.iloc[0]-1)*100):.1f}%"
        else:
            tendencia = f"QUEDA de {((1-y.iloc[-1]/y.iloc[0])*100):.1f}%"
        
        print(f"\nTendência Geral: {tendencia}")
        print(f"Valor Médio: {y.mean():.2f}")
        print(f"Volatilidade (CV): {(y.std()/y.mean()*100):.1f}%")
        
        # Melhor modelo
        if self.metricas:
            df_metricas = pd.DataFrame(self.metricas).T
            melhor_modelo = df_metricas['MAPE'].idxmin()
            melhor_mape = df_metricas.loc[melhor_modelo, 'MAPE']
            
            print(f"\n🏆 MODELO RECOMENDADO: {melhor_modelo}")
            print(f"   Precisão: {100-melhor_mape:.1f}% (Erro médio: {melhor_mape:.1f}%)")
            print(f"   Interpretação: O modelo erra em média {melhor_mape:.1f}% do valor real")
            
            # Próximas previsões
            if melhor_modelo in self.previsoes:
                prox_valor = self.previsoes[melhor_modelo].iloc[0]
                print(f"\nPRÓXIMA PREVISÃO: {prox_valor:.2f}")
                
                if f'{melhor_modelo}_lower' in self.previsoes:
                    lower = self.previsoes[f'{melhor_modelo}_lower'].iloc[0]
                    upper = self.previsoes[f'{melhor_modelo}_upper'].iloc[0]
                    print(f"   Intervalo de Confiança 95%: [{lower:.2f}, {upper:.2f}]")
                    print(f"   Cenário Pessimista: {lower:.2f}")
                    print(f"   Cenário Mais Provável: {prox_valor:.2f}")
                    print(f"   Cenário Otimista: {upper:.2f}")

    def dashboard_completo(self):
        """Dashboard executivo com KPIs principais."""
        y = self.df[self.nome_valor]
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # KPI 1: Valor Atual
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.6, f"{y.iloc[-1]:.2f}", 
                ha='center', va='center', fontsize=40, fontweight='bold', color='#2E86AB')
        ax1.text(0.5, 0.3, "VALOR ATUAL", 
                ha='center', va='center', fontsize=14, color='gray')
        variacao = ((y.iloc[-1] - y.iloc[-2]) / y.iloc[-2] * 100) if len(y) > 1 else 0
        cor_variacao = '#06FFA5' if variacao >= 0 else '#E63946'
        ax1.text(0.5, 0.15, f"{variacao:+.1f}%", 
                ha='center', va='center', fontsize=16, color=cor_variacao, fontweight='bold')
        ax1.axis('off')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # KPI 2: Média
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.6, f"{y.mean():.2f}", 
                ha='center', va='center', fontsize=40, fontweight='bold', color='#F4A261')
        ax2.text(0.5, 0.3, "MÉDIA HISTÓRICA", 
                ha='center', va='center', fontsize=14, color='gray')
        ax2.axis('off')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # KPI 3: Melhor Modelo
        ax3 = fig.add_subplot(gs[0, 2])
        if self.metricas:
            df_metricas = pd.DataFrame(self.metricas).T
            melhor = df_metricas['MAPE'].idxmin()
            mape = df_metricas.loc[melhor, 'MAPE']
            ax3.text(0.5, 0.6, melhor, 
                    ha='center', va='center', fontsize=28, fontweight='bold', color='#A23B72')
            ax3.text(0.5, 0.3, "MODELO RECOMENDADO", 
                    ha='center', va='center', fontsize=14, color='gray')
            ax3.text(0.5, 0.15, f"Erro: {mape:.1f}%", 
                    ha='center', va='center', fontsize=16, color='#E63946', fontweight='bold')
        ax3.axis('off')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # Gráfico 4: Série temporal com previsões
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(self.y_train.index, self.y_train, label='Histórico', 
                linewidth=2.5, color='black')
        ax4.plot(self.y_test.index, self.y_test, label='Real (Teste)', 
                linewidth=2.5, color='black', linestyle='--', marker='o')
        
        if self.metricas and melhor in self.previsoes:
            ax4.plot(self.previsoes[melhor].index, self.previsoes[melhor], 
                    label=f'Previsão ({melhor})', linewidth=3, color='#E63946', marker='s')
            
            if f'{melhor}_lower' in self.previsoes:
                ax4.fill_between(self.previsoes[melhor].index,
                               self.previsoes[f'{melhor}_lower'],
                               self.previsoes[f'{melhor}_upper'],
                               alpha=0.2, color='#E63946', label='IC 95%')
        
        ax4.axvline(self.y_train.index[-1], color='red', linestyle=':', linewidth=2, alpha=0.5)
        ax4.set_title('Série Temporal e Previsões', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Valor')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Distribuição
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(y, bins=25, density=True, alpha=0.7, color='#4ECDC4', edgecolor='black')
        ax5.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label='Média')
        ax5.axvline(y.median(), color='green', linestyle='--', linewidth=2, label='Mediana')
        ax5.set_title('Distribuição dos Valores', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Valor')
        ax5.set_ylabel('Densidade')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Métricas comparadas
        ax6 = fig.add_subplot(gs[2, 1:])
        if self.metricas:
            df_plot = df_metricas[['MAE', 'RMSE', 'MAPE']].sort_values('MAPE')
            x = np.arange(len(df_plot))
            width = 0.25
            
            ax6.bar(x - width, df_plot['MAE'], width, label='MAE', color='#E63946')
            ax6.bar(x, df_plot['RMSE'], width, label='RMSE', color='#F4A261')
            ax6.bar(x + width, df_plot['MAPE'], width, label='MAPE', color='#2A9D8F')
            
            ax6.set_xlabel('Modelos', fontweight='bold')
            ax6.set_ylabel('Erro', fontweight='bold')
            ax6.set_title('Comparação de Erros por Modelo', fontsize=12, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(df_plot.index, rotation=45, ha='right')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'DASHBOARD EXECUTIVO - {self.nome_valor}', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.show()

    # ==================== ANÁLISES ADICIONAIS ====================
    
    def analise_intervencao(self, data_intervencao):
        """Analisa o impacto de uma intervenção em uma data específica."""
        print(f"\n{'='*80}")
        print("ANÁLISE DE INTERVENÇÃO")
        print(f"{'='*80}\n")
        
        y = self.df[self.nome_valor]
        data_intervencao = pd.to_datetime(data_intervencao)
        
        # Dividir antes e depois
        antes = y[y.index < data_intervencao]
        depois = y[y.index >= data_intervencao]
        
        print(f"Data da Intervenção: {data_intervencao.strftime('%Y-%m-%d')}")
        print(f"\nPeríodo ANTES:")
        print(f"  Média: {antes.mean():.2f}")
        print(f"  Desvio: {antes.std():.2f}")
        print(f"  Observações: {len(antes)}")
        
        print(f"\nPeríodo DEPOIS:")
        print(f"  Média: {depois.mean():.2f}")
        print(f"  Desvio: {depois.std():.2f}")
        print(f"  Observações: {len(depois)}")
        
        # Teste t para diferença de médias
        t_stat, p_valor = stats.ttest_ind(antes, depois)
        diferenca_pct = ((depois.mean() - antes.mean()) / antes.mean()) * 100
        
        print(f"\nTESTE DE HIPÓTESE (t-test)")
        print(f"Estatística t: {t_stat:.4f}")
        print(f"P-valor: {p_valor:.4f}")
        print(f"Diferença: {diferenca_pct:+.1f}%")
        
        if p_valor < 0.05:
            print(f"✓ Impacto SIGNIFICATIVO detectado (p < 0.05)")
        else:
            print(f"✗ Impacto NÃO significativo (p >= 0.05)")
        
        # Visualização
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Série temporal
        axes[0].plot(y.index, y, linewidth=2, color='#2E86AB', label='Série')
        axes[0].axvline(data_intervencao, color='red', linestyle='--', 
                       linewidth=3, label='Intervenção')
        axes[0].axhline(antes.mean(), color='green', linestyle=':', 
                       linewidth=2, alpha=0.7, label=f'Média Antes: {antes.mean():.2f}')
        axes[0].axhline(depois.mean(), color='orange', linestyle=':', 
                       linewidth=2, alpha=0.7, label=f'Média Depois: {depois.mean():.2f}')
        axes[0].set_title('Análise de Intervenção', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Valor')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Boxplots comparativos
        axes[1].boxplot([antes, depois], labels=['Antes', 'Depois'], 
                       patch_artist=True,
                       boxprops=dict(facecolor='#FFE66D', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_title('Distribuição Antes vs Depois', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Valor')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

    def validacao_cruzada_temporal(self, n_splits=5):
        """Validação cruzada respeitando a ordem temporal."""
        print(f"\n{'='*80}")
        print("VALIDAÇÃO CRUZADA TEMPORAL")
        print(f"{'='*80}\n")
        
        y = self.df[self.nome_valor]
        n = len(y)
        tamanho_teste = n // (n_splits + 1)
        
        resultados = {modelo: [] for modelo in ['RandomForest', 'SARIMA', 'HoltWinters', 'Prophet']}
        
        print(f"Realizando {n_splits} splits temporais...")
        print(f"Tamanho do teste por split: {tamanho_teste} observações\n")
        
        for i in range(n_splits):
            split_point = n - (n_splits - i) * tamanho_teste
            train = y.iloc[:split_point]
            test = y.iloc[split_point:split_point + tamanho_teste]
            
            print(f"Split {i+1}/{n_splits}: Treino até {train.index[-1].strftime('%Y-%m-%d')}")
            
            # Random Forest
            try:
                def criar_features_cv(data, n_lags=12):
                    df = pd.DataFrame(index=data.index)
                    df['valor'] = data.values
                    for lag in range(1, n_lags + 1):
                        df[f'lag_{lag}'] = data.shift(lag)
                    for window in [3, 6, 12]:
                        df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
                        df[f'rolling_std_{window}'] = data.rolling(window=window).std()
                    df['mes'] = data.index.month
                    df['trimestre'] = data.index.quarter
                    df['ano'] = data.index.year
                    df['tendencia'] = np.arange(len(data))
                    return df.dropna()
                
                df_train_rf = criar_features_cv(train)
                X_train_rf = df_train_rf.drop('valor', axis=1)
                y_train_rf = df_train_rf['valor']
                
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train_rf, y_train_rf)
                
                predicoes_rf = []
                ultimo_treino = train.copy()
                for j in range(len(test)):
                    df_pred = criar_features_cv(ultimo_treino)
                    X_pred = df_pred.iloc[[-1]].drop('valor', axis=1)
                    pred_val = rf.predict(X_pred)[0]
                    predicoes_rf.append(pred_val)
                    novo_index = pd.DatetimeIndex([test.index[j]])
                    novo_valor = pd.Series([test.iloc[j]], index=novo_index)
                    ultimo_treino = pd.concat([ultimo_treino, novo_valor])
                
                mape = mean_absolute_percentage_error(test, predicoes_rf) * 100
                resultados['RandomForest'].append(mape)
            except:
                pass
            
            # SARIMA
            try:
                modelo = auto_arima(train, seasonal=True, m=12, trace=False, 
                                   error_action='ignore', suppress_warnings=True, 
                                   stepwise=True, max_p=3, max_q=3)
                pred = modelo.predict(n_periods=len(test))
                mape = mean_absolute_percentage_error(test, pred) * 100
                resultados['SARIMA'].append(mape)
            except:
                pass
            
            # Holt-Winters
            try:
                modelo = ExponentialSmoothing(train, trend='add', seasonal='add', 
                                             seasonal_periods=12).fit()
                pred = modelo.forecast(len(test))
                mape = mean_absolute_percentage_error(test, pred) * 100
                resultados['HoltWinters'].append(mape)
            except:
                pass
            
            # Prophet
            try:
                df_train = train.reset_index()
                df_train.columns = ['ds', 'y']
                modelo = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                               daily_seasonality=False).fit(df_train)
                future = modelo.make_future_dataframe(periods=len(test), freq='MS')
                forecast = modelo.predict(future)
                pred = forecast.tail(len(test))['yhat'].values
                mape = mean_absolute_percentage_error(test, pred) * 100
                resultados['Prophet'].append(mape)
            except:
                pass
            # Resumo
            print("\nRESULTADOS DA VALIDAÇÃO CRUZADA")
            print("=" * 60)
            for modelo, erros in resultados.items():
                if erros:
                    print(f"{modelo:.<30} MAPE Médio: {np.mean(erros):.2f}% (±{np.std(erros):.2f}%)")
            print("=" * 60 + "\n")
            
            # Visualização
            fig, ax = plt.subplots(figsize=(12, 6))
            data_plot = [erros for erros in resultados.values() if erros]
            labels = [modelo for modelo, erros in resultados.items() if erros]
            
            bp = ax.boxplot(data_plot, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='#4ECDC4', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
            
            ax.set_ylabel('MAPE (%)', fontweight='bold')
            ax.set_title('Validação Cruzada Temporal - Distribuição de Erros', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()

    def analise_completa(self, periodos_futuros=12):
        """
        Executa pipeline completo de análise.
        
        Args:
            periodos_futuros: Número de períodos futuros para prever (default: 12)
        """
        print("\n" + "="*80)
        print("INICIANDO ANÁLISE COMPLETA DE SÉRIE TEMPORAL")
        print("="*80)
        
        # 1. Exploração
        self.analise_exploratoria_completa()
        
        # 2. Estacionariedade
        self.testes_estacionariedade()
        
        # 3. Modelagem
        self.ajustar_todos_modelos()
        
        # 4. Previsões Futuras
        previsoes_futuras = self.prever_futuro(periodos_futuros)
        
        # 5. Dashboard
        self.dashboard_completo()
        
        # 6. Relatório
        self.gerar_relatorio_executivo()
        
        print("\n" + "="*80)
        print("ANÁLISE COMPLETA FINALIZADA")
        print("="*80 + "\n")
        
        return previsoes_futuras