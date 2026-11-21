import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

from scipy.optimize import differential_evolution
import pyswarms as ps


# classe que herd do MLPRegressor e força a inicialização dos pesos entre [-1, 1] numa uniforme
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
    data = df[value_col].copy()

    result = pd.DataFrame()

    for i in range(k, 0, -1):
        result[f'x-{i}'] = data.shift(i)

    result['x'] = data

    result = result.dropna().reset_index(drop=True)

    return result



def create_fitness(df, value_col, date_col='dt', plot=False):

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

        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

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
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()

        predictions_train = model.predict(X_train)
        predictions_train = scaler_y.inverse_transform(predictions_train.reshape(-1, 1)).ravel()

        if plot:
            datas = df[date_col].values[k:][split:]

            plt.figure(figsize=(10, 5))
            plt.plot(datas, predictions, label='Valores previstos')
            plt.plot(datas, y_test, label='Valores reais')

            plt.title('Valores reais vs. previstos')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()

        return {
            'test_error' : root_mean_squared_error(y_test, predictions), 
            'train_error' : root_mean_squared_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel(), predictions_train)
        }

    return fitness


def PSO_lbest(fitness, n_particles=30, iters=200, ftol_iter=20, c1=1.49618, c2=1.49618, w=0.7298, k=3, p=2, seed=42, plot=False):
    # Função objetivo: recebe todas as partículas
    def objective_function(particles):

        costs = []

        for particle in particles:
            cost = fitness(particle)['test_error']
            costs.append(cost)

        return np.array(costs)


    # Espaço de busca
    lb = np.array([3, 1, 0.001])
    ub = np.array([10, 50, 0.1])

    bounds = (lb, ub)


    # Parâmetros do LBest PSO
    options = {
        'c1': c1,
        'c2': c2,
        'w': w,
        'k': k,  # Número de vizinhos
        'p': p   # Distância euclidiana
    }


    # Reprodutibilidade
    np.random.seed(seed)


    # Inicialização
    optimizer = ps.single.LocalBestPSO(
        n_particles=n_particles,
        dimensions=3,
        options=options,
        bounds=bounds,
        ftol=1e-6,
        ftol_iter=ftol_iter
    )


    # Otimização
    best_cost, best_position = optimizer.optimize(
        objective_function,
        iters=iters
    )

    # Melhores parâmetros
    best_k = int(round(best_position[0]))
    best_n = int(round(best_position[1]))
    best_a = best_position[2]
    
    if plot:

        # Convergência
        plt.figure(figsize=(10, 5))

        plt.plot(optimizer.cost_history)

        plt.xlabel('Iteração')
        plt.ylabel('Melhor MSE global encontrado')
        plt.title('Convergência do LBest PSO')

        plt.grid(alpha=0.3)
        plt.show()

    return best_cost, best_k, best_n, best_a


def DE_DEBEST1(fitness, pop_size=5, iters=50, mutation=(0.6, 0.9), recombination=0.7, seed=42, plot=False, patience=20, min_delta=1e-2):
    def objective_function(params):
        return fitness(params)['test_error']

    bounds = [
        (3, 10),
        (1, 50),
        (0.001, 0.1)
    ]

    history = []

    state = {
        "best": None,
        "stagnant": 0
    }

    def callback(intermediate_result):
        current_best = intermediate_result.fun
        history.append(current_best)

        if state["best"] is None:
            state["best"] = current_best
            state["stagnant"] = 0

        elif current_best < state["best"] - (min_delta * max(abs(state["best"]), 1e-12)):
            state["best"] = current_best
            state["stagnant"] = 0

        else:
            state["stagnant"] += 1

        return state["stagnant"] >= patience

    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        strategy='best1bin',
        maxiter=iters,
        popsize=pop_size,
        mutation=mutation,
        recombination=recombination,
        tol=1e-2,
        atol=0,
        integrality=[True, True, False],
        polish=False,
        seed=seed,
        disp=True,
        callback=callback
    )

    best_k = int(result.x[0])
    best_n = int(result.x[1])
    best_a = result.x[2]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(history)
        plt.xlabel('Geração')
        plt.ylabel('Melhor fitness')
        plt.title('Convergência do DE/best/1/bin')
        plt.grid(alpha=0.3)
        plt.show()

    return (
        result.fun,
        best_k,
        best_n,
        best_a,
    )


def run_experiments(algorithm, df, dataset_name, configs, n_runs=10, base_seed=2026, value_col="LandAverageTemperature", output_dir="resultados"):
    algorithm = algorithm.lower()

    # Função objetivo da base
    fitness = create_fitness(df, value_col=value_col)

    # Sementes reproduzíveis e distintas
    seeds = [base_seed + i for i in range(n_runs)]

    results = []

    for config_id, config in enumerate(configs, start=1):

        print(
            f"\n{algorithm.upper()} | {dataset_name} | "
            f"Configuração {config_id}/{len(configs)}"
        )

        for run_id, seed in enumerate(seeds, start=1):

            print(f"Execução {run_id}/{n_runs} | seed={seed}")

            start_time = time.perf_counter()

            if algorithm == "pso":
                best_cost, best_k, best_n, best_a = PSO_lbest(fitness, seed=seed, plot=False, **config)

            else:
                best_cost, best_k, best_n, best_a = DE_DEBEST1(fitness, seed=seed, plot=False, **config)

            elapsed_time = time.perf_counter() - start_time

            results.append({
                "algorithm": algorithm.upper(),
                "dataset": dataset_name,
                "config_id": config_id,
                "run": run_id,
                "seed": seed,
                **config,
                "best_rmse": best_cost,
                "best_k": best_k,
                "best_n": best_n,
                "best_learning_rate": best_a,
                "time_seconds": elapsed_time
            })

    # Organiza os resultados
    results_df = pd.DataFrame(results)

    # Cria diretório, caso não exista
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Um CSV por algoritmo e base
    filename = f"{algorithm}_{dataset_name}.csv"
    filepath = output_path / filename

    results_df.to_csv(filepath, index=False, encoding="utf-8-sig")

    print(f"\nResultados salvos em: {filepath}")

    return results_df
