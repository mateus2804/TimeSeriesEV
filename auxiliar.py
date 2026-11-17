import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import root_mean_squared_error


class MLPRegressorUniform(MLPRegressor):

    def _init_coef(self, fan_in, fan_out, dtype):

        # Pesos da camada
        coef_init = self._random_state.uniform(
            -1, 1, (fan_in, fan_out)
        ).astype(dtype)

        # Biases da camada
        intercept_init = self._random_state.uniform(
            -1, 1, fan_out
        ).astype(dtype)

        return coef_init, intercept_init


def create_lags(df, k, value_col):
    """
    Transforma a série temporal em um DataFrame supervisionado.

    k: quantidade de lags
    value_col: nome da coluna com os valores numéricos
    """

    data = df[value_col].copy()

    result = pd.DataFrame()

    for i in range(k, 0, -1):
        result[f'x-{i}'] = data.shift(i)

    result['x'] = data

    result = result.dropna().reset_index(drop=True)

    return result



def create_fitness(df, value_col, date_col='dt'):

    def fitness(params):

        k = int(round(params[0]))
        n = int(round(params[1]))
        a = params[2]

        if k < 1 or n < 1:
            return 1e10

        data = create_lags(df, k, value_col)

        X = data.drop(columns=['x']).values
        y = data['x'].values

        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Normalização
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        y_train = scaler_y.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()

        # MLP
        model = MLPRegressorUniform(
            hidden_layer_sizes=(n,),
            activation='relu',
            solver='adam',
            learning_rate_init=a,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=30,
            tol=1e-3
        )

        model.fit(X_train, y_train)

        # Previsão
        predictions = model.predict(X_test)

        predictions = scaler_y.inverse_transform(
            predictions.reshape(-1, 1)
        ).ravel()

        return root_mean_squared_error(y_test, predictions)

    return fitness
