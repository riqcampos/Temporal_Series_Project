import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import logging

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

class UniversalForecaster:
    def __init__(self, df: pd.DataFrame, date_col: str, value_col: str, freq: str = 'D'):
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        self.freq = freq
        
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values(by=self.date_col).set_index(self.date_col)
        # Correção do fillna
        self.df = self.df.asfreq(self.freq).ffill()
        
        self.best_model_name = None
        self.best_model_instance = None # Armazena o objeto treinado real
        self.best_model_residuals = None # O elo perdido para análise de ruído
        self.forecast_results = {}

    def _create_lagged_features(self, series, lags=5):
        df_lag = pd.DataFrame(series)
        for i in range(1, lags + 1):
            df_lag[f'lag_{i}'] = df_lag[self.value_col].shift(i)
        return df_lag.dropna()

    def _evaluate_model_cv(self, model_type, n_splits=3):
        # Ajuste dinâmico de splits para dados pequenos
        if len(self.df) < 20:
            n_splits = 2
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        errors = []
        
        y = self.df[self.value_col]

        for train_index, test_index in tscv.split(y):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            try:
                preds = []
                
                if model_type == 'HoltWinters':
                    # Ajuste de sazonalidade se dados forem insuficientes
                    seasonal_periods = 12 if self.freq == 'MS' else 7
                    if len(y_train) < 2 * seasonal_periods:
                        model = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
                    else:
                        model = ExponentialSmoothing(y_train, seasonal='add', trend='add', seasonal_periods=seasonal_periods).fit()
                    preds = model.forecast(len(y_test))

                elif model_type == 'SARIMA':
                    model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    preds = model.forecast(steps=len(y_test))

                elif model_type == 'Prophet' and PROPHET_AVAILABLE:
                    df_p = y_train.reset_index()
                    df_p.columns = ['ds', 'y']
                    m = Prophet(daily_seasonality=True if self.freq=='D' else False)
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=len(y_test), freq=self.freq)
                    preds = m.predict(future)['yhat'].tail(len(y_test)).values

                elif model_type == 'DecisionTree':
                    lags = min(12, len(y_train)//2) # Lags dinâmicos
                    df_lags = self._create_lagged_features(self.df, lags=lags)
                    train_lags = df_lags.loc[y_train.index.intersection(df_lags.index)]
                    test_lags = df_lags.loc[y_test.index.intersection(df_lags.index)]
                    
                    if len(train_lags) > 0 and len(test_lags) > 0:
                        reg = DecisionTreeRegressor(max_depth=5, random_state=42)
                        reg.fit(train_lags.drop(columns=[self.value_col]), train_lags[self.value_col])
                        preds = reg.predict(test_lags.drop(columns=[self.value_col]))
                        y_test = y_test.loc[test_lags.index]

                if len(preds) == len(y_test):
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    errors.append(rmse)
            
            except Exception as e:
                continue

        return np.mean(errors) if errors else np.inf

    def fit_and_select_best(self):
        print("\n--- Batalha de Modelos (AutoML) ---")
        models = ['HoltWinters', 'SARIMA', 'DecisionTree']
        if PROPHET_AVAILABLE: models.append('Prophet')

        results = {}
        for m in models:
            score = self._evaluate_model_cv(m)
            if score != np.inf:
                print(f"Modelo: {m:<15} | RMSE CV: {score:.4f}")
                results[m] = score
            else:
                print(f"Modelo: {m:<15} | Falhou na validação")
        
        if not results:
            raise ValueError("Nenhum modelo conseguiu treinar com estes dados.")

        self.best_model_name = min(results, key=results.get)
        print(f">>> VENCEDOR: {self.best_model_name}")
        return self.results_cv

    def predict_future(self, steps=12):
        """
        Retreina o melhor modelo com TODOS os dados e captura os resíduos para análise posterior.
        """
        y_all = self.df[self.value_col]
        forecast_values = []
        residuals = None

        print(f"Treinando {self.best_model_name} em todo o histórico...")

        try:
            if self.best_model_name == 'HoltWinters':
                per = 12 if self.freq == 'MS' else 7
                seasonal = 'add' if len(y_all) > 2*per else None
                model = ExponentialSmoothing(y_all, seasonal=seasonal, trend='add', seasonal_periods=per).fit()
                forecast_values = model.forecast(steps)
                # Captura resíduos (Observado - Ajustado)
                self.best_model_residuals = y_all - model.fittedvalues

            elif self.best_model_name == 'SARIMA':
                model = SARIMAX(y_all, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
                forecast_values = model.get_forecast(steps=steps).predicted_mean
                self.best_model_residuals = model.resid

            elif self.best_model_name == 'Prophet':
                df_p = y_all.reset_index()
                df_p.columns = ['ds', 'y']
                m = Prophet(daily_seasonality=False)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=steps, freq=self.freq)
                fcst = m.predict(future)
                forecast_values = fcst['yhat'].tail(steps)
                forecast_values.index = pd.date_range(start=y_all.index[-1], periods=steps+1, freq=self.freq)[1:]
                # Resíduos no Prophet
                self.best_model_residuals = y_all.values - fcst['yhat'].iloc[:-steps].values

            elif self.best_model_name == 'DecisionTree':
                lags = min(12, len(y_all)//2)
                df_lags = self._create_lagged_features(self.df, lags=lags)
                reg = DecisionTreeRegressor(max_depth=5).fit(df_lags.drop(columns=[self.value_col]), df_lags[self.value_col])
                
                # Recursivo simples
                curr = df_lags.iloc[-1].drop(columns=[self.value_col]).values
                preds = []
                for _ in range(steps):
                    p = reg.predict([curr])[0]
                    preds.append(p)
                    curr = np.roll(curr, 1)
                    curr[0] = p
                
                forecast_values = pd.Series(preds, index=pd.date_range(start=y_all.index[-1], periods=steps+1, freq=self.freq)[1:])
                # Resíduos ML (Observado - Predito no treino)
                fitted = reg.predict(df_lags.drop(columns=[self.value_col]))
                self.best_model_residuals = df_lags[self.value_col] - fitted

            # Salva no objeto
            self.forecast_results = {
                "forecast": forecast_values,
                "history": y_all
            }
            
            # Garante que resíduos sejam Series com índice temporal se possível
            if not isinstance(self.best_model_residuals, pd.Series):
                # Tenta alinhar pelo fim se tamanhos divergirem (comum em ML com lags)
                idx = y_all.index[-len(self.best_model_residuals):]
                self.best_model_residuals = pd.Series(self.best_model_residuals, index=idx)

            return self.forecast_results
            
        except Exception as e:
            print(f"Erro fatal na predição: {e}")
            return None