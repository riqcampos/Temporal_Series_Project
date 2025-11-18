import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import logging

# Tenta importar o Prophet (biblioteca externa comum no seu escopo)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class UniversalForecaster:
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str, freq: str = 'D'):
        """
        Inicializa o orquestrador de previsões.
        
        :param freq: Frequência da série (ex: 'D', 'MS', 'H'). Vital para modelos estatísticos.
        """
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        self.freq = freq
        
        # Preparação
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(by=self.date_col).set_index(self.date_col)
        self.df = self.df.asfreq(self.freq).fillna(method='ffill') # Preenchimento simples para evitar erros
        
        self.results_cv = {}
        self.best_model_name = None
        self.best_model_instance = None
        self.forecast_results = {}

    def _create_lagged_features(self, series, lags=5):
        """Gera features de atraso para modelos de ML (Decision Tree)."""
        df_lag = pd.DataFrame(series)
        for i in range(1, lags + 1):
            df_lag[f'lag_{i}'] = df_lag[self.value_col].shift(i)
        return df_lag.dropna()

    def _evaluate_model_cv(self, model_type, n_splits=3):
        """
        Executa Validação Cruzada em Série Temporal (Walk-Forward Validation).
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        errors = []
        
        X = self.df.index
        y = self.df[self.value_col]

        for train_index, test_index in tscv.split(y):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            try:
                preds = []
                
                # MODELO 1: HOLT-WINTERS (Suavização Exponencial)
                if model_type == 'HoltWinters':
                    model = ExponentialSmoothing(
                        y_train, 
                        seasonal='add', 
                        trend='add', 
                        seasonal_periods=12 if self.freq == 'MS' else 7
                    ).fit()
                    preds = model.forecast(len(y_test))

                # MODELO 2: SARIMA (Simplificado)
                elif model_type == 'SARIMA':
                    # Ordem fixa para generalização (ideal seria auto_arima, mas é lento para CV)
                    # (1, 1, 1) x (1, 1, 0, 12) é um "chute" estatístico seguro para muitas séries
                    model = SARIMAX(
                        y_train, 
                        order=(1, 1, 1), 
                        seasonal_order=(0, 1, 1, 12 if self.freq == 'MS' else 7),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)
                    preds = model.forecast(steps=len(y_test))

                # --- MODELO 3: PROPHET ---
                elif model_type == 'Prophet' and PROPHET_AVAILABLE:
                    df_prophet = y_train.reset_index()
                    df_prophet.columns = ['ds', 'y']
                    m = Prophet(daily_seasonality=True if self.freq=='D' else False)
                    m.fit(df_prophet)
                    future = m.make_future_dataframe(periods=len(y_test), freq=self.freq)
                    forecast = m.predict(future)
                    preds = forecast['yhat'].tail(len(y_test)).values

                # --- MODELO 4: DECISION TREE (ML) ---
                elif model_type == 'DecisionTree':
                    # Feature Engineering on the fly
                    df_lags = self._create_lagged_features(self.df, lags=12)
                    
                    # Split based on indices respecting lags
                    train_lags = df_lags.loc[y_train.index.intersection(df_lags.index)]
                    test_lags = df_lags.loc[y_test.index.intersection(df_lags.index)]
                    
                    if len(train_lags) == 0 or len(test_lags) == 0:
                        continue
                        
                    reg = DecisionTreeRegressor(max_depth=10, random_state=42)
                    reg.fit(train_lags.drop(columns=[self.value_col]), train_lags[self.value_col])
                    preds = reg.predict(test_lags.drop(columns=[self.value_col]))
                    # Ajuste de tamanho caso lags tenham cortado dados
                    y_test = y_test.loc[test_lags.index]

                # Cálculo de erro
                if len(preds) == len(y_test):
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    errors.append(rmse)
            
            except Exception as e:
                # print(f"Erro no modelo {model_type}: {str(e)}")
                continue

        return np.mean(errors) if errors else np.inf

    def fit_and_select_best(self):
        """Roda a competição de modelos e define o vencedor."""
        print("Iniciando competição de modelos com Validação Cruzada...")
        
        models = ['HoltWinters', 'SARIMA', 'DecisionTree']
        if PROPHET_AVAILABLE:
            models.append('Prophet')

        results = {}
        for m in models:
            print(f"Testando: {m}...")
            score = self._evaluate_model_cv(m)
            results[m] = score
            
        self.results_cv = results
        
        # Seleciona o melhor (menor RMSE)
        self.best_model_name = min(results, key=results.get)
        print(f"\n>>> VENCEDOR: {self.best_model_name} (RMSE Médio: {results[self.best_model_name]:.2f})")
        return self.results_cv

    def predict_future(self, steps=30):
        """
        Treina o modelo vencedor em TODOS os dados e projeta o futuro.
        """
        if not self.best_model_name:
            raise ValueError("Execute 'fit_and_select_best' primeiro.")

        print(f"Gerando previsão futura de {steps} passos com {self.best_model_name}...")
        
        y_all = self.df[self.value_col]
        forecast_values = []
        conf_int = None # Para modelos que suportam

        # Retreino Full
        if self.best_model_name == 'HoltWinters':
            model = ExponentialSmoothing(
                y_all, seasonal='add', trend='add', 
                seasonal_periods=12 if self.freq == 'MS' else 7
            ).fit()
            forecast_values = model.forecast(steps)

        elif self.best_model_name == 'SARIMA':
            model = SARIMAX(
                y_all, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12 if self.freq == 'MS' else 7)
            ).fit(disp=False)
            forecast = model.get_forecast(steps=steps)
            forecast_values = forecast.predicted_mean
            conf_int = forecast.conf_int()

        elif self.best_model_name == 'Prophet':
            df_prophet = y_all.reset_index()
            df_prophet.columns = ['ds', 'y']
            m = Prophet(daily_seasonality=True if self.freq=='D' else False)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=steps, freq=self.freq)
            fcst = m.predict(future)
            forecast_values = fcst['yhat'].tail(steps)
            forecast_values.index = pd.date_range(start=y_all.index[-1], periods=steps+1, freq=self.freq)[1:]
            
        elif self.best_model_name == 'DecisionTree':
            # Estratégia recursiva para ML puro
            df_lags = self._create_lagged_features(self.df, lags=12)
            reg = DecisionTreeRegressor(max_depth=10, random_state=42)
            reg.fit(df_lags.drop(columns=[self.value_col]), df_lags[self.value_col])
            
            # Previsão passo a passo (recursiva)
            curr_features = df_lags.iloc[-1].drop(columns=[self.value_col]).values
            preds = []
            for _ in range(steps):
                pred = reg.predict([curr_features])[0]
                preds.append(pred)
                # Atualiza features (shift para esquerda e insere nova predição)
                curr_features = np.roll(curr_features, 1)
                curr_features[0] = pred # lag_1 é o mais recente
            
            forecast_values = pd.Series(preds, index=pd.date_range(start=y_all.index[-1], periods=steps+1, freq=self.freq)[1:])

        self.forecast_results = {
            "model": self.best_model_name,
            "forecast": forecast_values,
            "history": y_all,
            "conf_int": conf_int
        }
        return self.forecast_results

    def plot_final_results(self):
        """Gera visualização executiva dos resultados."""
        if not self.forecast_results:
            return
            
        hist = self.forecast_results['history']
        pred = self.forecast_results['forecast']
        model_name = self.forecast_results['model']
        conf_int = self.forecast_results.get('conf_int')

        plt.figure(figsize=(14, 7))
        
        # Plota histórico recente
        plt.plot(hist.index[-int(len(hist)/2):], hist.values[-int(len(hist)/2):], label='Histórico Real', color='gray')
        
        # Plota Previsão
        plt.plot(pred.index, pred.values, label=f'Previsão ({model_name})', color='red', linestyle='--', linewidth=2)
        
        # Plota Intervalo de Confiança (se houver - SARIMA/Prophet)
        if conf_int is not None and not conf_int.empty:
            plt.fill_between(pred.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confiança 95%')

        plt.title(f"Previsão de Demanda - Modelo Vencedor: {model_name}", fontsize=16)
        plt.xlabel("Tempo")
        plt.ylabel("Valores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Plot de Performance Comparativa
        plt.figure(figsize=(10, 5))
        scores = pd.Series(self.results_cv).sort_values()
        sns.barplot(x=scores.values, y=scores.index, palette='viridis')
        plt.title("Erro Médio (RMSE) por Modelo na Validação Cruzada\n(Menor é melhor)")
        plt.xlabel("RMSE")
        plt.show()

# --- EXEMPLO DE USO ---
if __name__ == "__main__":
    # Criação de dados fictícios
    dates = pd.date_range(start='2020-01-01', periods=100, freq='MS')
    values = np.linspace(100, 200, 100) + np.random.normal(0, 10, 100) + (np.sin(np.arange(100)/4) * 20)
    df_teste = pd.DataFrame({'Data': dates, 'Vendas': values})

    # 1. Instancia
    forecaster = UniversalForecaster(df_teste, 'Data', 'Vendas', freq='MS')
    
    # 2. Seleciona Melhor Modelo
    metrics = forecaster.fit_and_select_best()
    
    # 3. Faz Previsão Futura (ex: 12 meses)
    resultado = forecaster.predict_future(steps=12)
    
    # 4. Gráficos
    forecaster.plot_final_results()