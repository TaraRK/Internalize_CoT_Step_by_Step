import time
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from .metrics import get_regression_perf_metrics
import gurobipy as gp
import numpy as np
from sklearn.feature_selection import f_regression, mutual_info_regression, r_regression
from sklearn.linear_model import LinearRegression, Lasso

import os 
os.environ['GRB_LICENSE_FILE'] = '/raid/lingo/carlguo/gurobi.lic'

def solve_inner_problem(X, Y, s, gamma):
    indices = np.where(s > 0.5)[0]
    n, d = X.shape
    denom = 2*n
    Xs = X[:, indices]

    alpha = Y - Xs @ (np.linalg.inv(np.eye(len(indices)) /
                      gamma + Xs.T @ Xs) @ (Xs.T @ Y))
    obj = np.dot(Y, alpha) / denom
    tmp = X.T @ alpha
    grad = -gamma * tmp**2 / denom
    return obj, grad


def get_heuristic_neuron_ranking_regression(X, y, method):
    if method == 'l1':
        lr = Lasso()
        lr = lr.fit(X, y)
        ranks = np.argsort(np.abs(lr.coef_[0]))
    elif method == 'f_stat':
        f_stat, p_val = f_regression(X, y)
        ranks = np.argsort(f_stat)
    elif method == 'mi':
        mi = mutual_info_regression(X, y)
        ranks = np.argsort(mi)

    elif method == 'correlation':
        corr = r_regression(X, y)
        ranks = np.argsort(np.abs(corr))
    else:
        raise ValueError('Invalid method')
    return ranks


def make_regression_k_list():
    base_ks = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14]
    exp_ks = list((2 ** np.linspace(4, 8, 13)).astype(int))
    return base_ks + exp_ks


def make_k_list(d_max, max_k=None):
    if max_k is None:
        max_k = d_max
    base_ks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    exp_ks = list((2 ** np.linspace(4, 8, 13)).astype(int))
    if exp_ks[-1] != d_max:
        exp_ks.append(d_max)
    return [k for k in base_ks + exp_ks if k <= max_k]


def dense_regression_probe(exp_cfg, activation_dataset, regression_target):
    """
    Train a dense probe on the activation dataset.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    regression_target : np.ndarray (n_samples) with regression targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activation_dataset, regression_target,
        test_size=exp_cfg.test_set_frac, random_state=exp_cfg.seed)

    lr = ElasticNet(precompute=True)

    start_t = time.time()
    lr = lr.fit(X_train, y_train)
    elapsed_time = time.time() - start_t 
    lr_pred = lr.predict(X_test)

    results = get_regression_perf_metrics(y_test, lr_pred)
    results['elapsed_time'] = elapsed_time
    results['n_iter'] = lr.n_iter_
    results['coef'] = lr.coef_
    return results


def heuristic_sparse_regression_sweep(activation_dataset, regression_target):
    """
    Train a heuristic sparse probe on the activation dataset for varying k.

    Parameters
    ----------
    exp_cfg : as specified by the CLI in probing_experiment.py
    activation_dataset : np.ndarray (n_samples, n_neurons)
    regression_target : np.ndarray (n_samples) with regression targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activation_dataset, regression_target,
        test_size=0.1, random_state=42)

    neuron_ranking = get_heuristic_neuron_ranking_regression(
        X_train, y_train, 'f_stat')

    layer_results = {}
    for k in make_regression_k_list()[::-1]:
        support = np.sort(neuron_ranking[-k:])
        lr = ElasticNet(precompute=True)
        start_t = time.time()
        lr = lr.fit(X_train[:, support], y_train)
        elapsed_time = time.time() - start_t

        lr_pred = lr.predict(X_test[:, support])
        layer_results[k] = get_regression_perf_metrics(y_test, lr_pred)
        layer_results[k]['elapsed_time'] = elapsed_time
        layer_results[k]['n_iter'] = lr.n_iter_
        layer_results[k]['coef'] = lr.coef_
        layer_results[k]['support'] = support

        # rerank according to the linear regression coefficients
        neuron_ranking = np.zeros(len(neuron_ranking))
        neuron_ranking[support] = np.abs(lr.coef_)
        neuron_ranking = np.argsort(neuron_ranking)

    return layer_results





def sparse_regression_oa(X, Y, k, gamma, s0, time_limit=60, verbose=True):
    n, d = X.shape

    gp_env = gp.Env()  # need env for cluster
    model = gp.Model("classifier", env=gp_env)

    s = model.addVars(d, vtype=gp.GRB.BINARY, name="support")
    t = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="objective")

    model.addConstr(gp.quicksum(s) <= k, name="l0")

    if len(s0) == 0:
        s0 = np.zeros(d)
        s0[range(int(k))] = 1

    obj0, grad0 = solve_inner_problem(X, Y, s0, gamma)
    model.addConstr(
        t >= obj0 + gp.quicksum(grad0[i] * (s[i] - s0[i]) for i in range(d)))
    model.setObjective(t, gp.GRB.MINIMIZE)

    def outer_approximation(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            s_bar = model.cbGetSolution(model._vars)
            s_vals = np.array([a for a in s_bar.values()])
            obj, grad = solve_inner_problem(X, Y, s_vals, gamma)
            model.cbLazy(
                t >= obj + gp.quicksum(grad[i] * (s[i] - s_vals[i]) for i in range(d)))

    model._vars = s
    model.params.OutputFlag = 1 if verbose else 0
    model.Params.lazyConstraints = 1
    model.Params.timeLimit = time_limit
    model.optimize(outer_approximation)

    support_indices = sorted([i for i in range(len(s)) if s[i].X > 0.5])

    X_s = X[:, support_indices]
    beta = np.zeros(d)
    sol = np.linalg.solve(np.eye(int(k)) / gamma + X_s.T @ X_s, X_s.T @ Y)
    beta[support_indices] = gamma * X_s.T @ (Y - X_s @ sol)

    model_stats = {
        'obj': model.ObjVal,
        'obj_bound': model.ObjBound,
        'mip_gap': model.MIPGap,
        'model_status': model.Status,
        'sol_count': model.SolCount,
        'iter_count': model.IterCount,
        'node_count': model.NodeCount,
        'runtime': model.Runtime
    }
    print(model_stats, type(model_stats), type(model))
    model.dispose()
    gp_env.dispose()

    return model_stats, beta, support_indices

def optimal_sparse_regression_probe(activation_dataset, regression_target, regularization=None, l1_ratio=0.5):
    """
    Train an optimal sparse probe for regression for various values of k (use telescoping for warm starts).

    Parameters
    ----------
    activation_dataset : np.ndarray (n_samples, n_neurons)
    regression_target : np.ndarray (n_samples) with regression targets
    regularization : float, optional defaults to 1/n**0.5
    l1_ratio : float, optional
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activation_dataset, regression_target,
        test_size=0.1, random_state=42)
    
    n, d_act = X_train.shape
    
    if regularization is None:
        regularization = 1/n**0.5
    
    scores = get_heuristic_neuron_ranking_regression(
        X_train, y_train, 'f_stat')
    
    # Score is correlation between each feature and target
    # scores = np.abs(np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] 
                            #  for i in range(d_act)]))
    # print("scores", scores)
    # Filter to top features based on correlation
    osp_heuristic_filter_size = 50
    coef_filter = np.sort(scores.argsort()[
        -osp_heuristic_filter_size:])
    scores = scores[coef_filter]
    
    osp_upto_k = 8
    ks = make_k_list(d_act, osp_upto_k)
    print("ks", ks)
    layer_results = {}
    for k in ks[::-1]:  # iterate in descending order
        # warm start - set max_k highest scores to 1
        s0 = np.zeros_like(coef_filter)
        s0[np.argsort(scores)[-k:]] = 1
        
        # Use sparse regression optimization
        model_stats, beta, support_indices = sparse_regression_oa(
            X_train[:, coef_filter], y_train, k,
            regularization, s0,
            time_limit=60, verbose=False
        )
        
        # Map support indices back to original feature space
        support = coef_filter[support_indices]
        
        # Get predictions using the sparse model
        X_test_s = X_test[:, support]
        y_pred = X_test_s @ beta[support_indices]
        
        # Calculate regression metrics
        osp_perf = get_regression_perf_metrics(y_test, y_pred)
        layer_results[('OSP', k)] = {**osp_perf, **model_stats}
        layer_results[('OSP', k)]['support'] = support
        layer_results[('OSP', k)]['beta'] = beta[support_indices]
        
        # Use recovered features to train an elastic net model
        k_elastic = ElasticNet(
            alpha=regularization,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-4,
            precompute=True
        )
        start_t = time.time()
        k_elastic = k_elastic.fit(X_train[:, support], y_train)
        k_elastic_elapsed_time = time.time() - start_t
        
        k_elastic_pred = k_elastic.predict(X_test[:, support])
        layer_results[('ElasticNet', k)] = get_regression_perf_metrics(
            y_test, k_elastic_pred)
        layer_results[('ElasticNet', k)]['runtime'] = k_elastic_elapsed_time
        layer_results[('ElasticNet', k)]['support'] = support
        layer_results[('ElasticNet', k)]['beta'] = k_elastic.coef_
        layer_results[('ElasticNet', k)]['bias'] = k_elastic.intercept_
        
        # Update scores based on elastic net coefficients
        scores = np.zeros_like(scores)
        scores[support_indices] = np.abs(k_elastic.coef_)
        
    return layer_results