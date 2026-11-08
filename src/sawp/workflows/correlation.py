from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _aligned(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    return pd.concat([left.rename("x"), right.rename("y")], axis=1).dropna()


def pairwise_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson, Spearman and Kendall statistics for every variable pair."""
    from scipy.stats import kendalltau, pearsonr, spearmanr

    rows = []
    for left_name, right_name in itertools.combinations(frame.columns, 2):
        pair = _aligned(frame[left_name], frame[right_name])
        if len(pair) < 3:
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "n": len(pair),
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "kendall_tau": np.nan,
                    "kendall_p": np.nan,
                }
            )
            continue
        pearson = pearsonr(pair["x"], pair["y"])
        spearman = spearmanr(pair["x"], pair["y"])
        kendall = kendalltau(pair["x"], pair["y"])
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "n": len(pair),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "kendall_tau": kendall.statistic,
                "kendall_p": kendall.pvalue,
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(
    frame: pd.DataFrame,
    *,
    method: str = "pearson",
    minimum_periods: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a correlation matrix and a matching two-sided p-value matrix."""
    from scipy.stats import kendalltau, pearsonr, spearmanr

    functions = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}
    if method not in functions:
        raise ValueError(f"unsupported correlation method: {method}")
    columns = list(frame.columns)
    coefficients = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    p_values = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    function = functions[method]
    for left_index, left_name in enumerate(columns):
        for right_index in range(left_index + 1, len(columns)):
            right_name = columns[right_index]
            pair = _aligned(frame[left_name], frame[right_name])
            if len(pair) < minimum_periods:
                coefficient, p_value = np.nan, np.nan
            else:
                result = function(pair["x"], pair["y"])
                coefficient, p_value = result.statistic, result.pvalue
            coefficients.iloc[left_index, right_index] = coefficient
            coefficients.iloc[right_index, left_index] = coefficient
            p_values.iloc[left_index, right_index] = p_value
            p_values.iloc[right_index, left_index] = p_value
    return coefficients, p_values
