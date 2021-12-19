

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.utils.extmath import stable_cumsum


def auuc(y_true, uplift, treatment):
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline

    return auc_score_actual / auc_score_perfect

def perfect_uplift_curve(y_true, treatment):

    y_true, treatment = np.array(y_true), np.array(treatment)

    cr_num = np.sum((y_true == 1) & (treatment == 0))  # Control Responders,
    tn_num = np.sum((y_true == 0) & (treatment == 1))  # Treated Non-Responders,

    # express an ideal uplift curve through y_true and treatment
    summand = y_true if cr_num > tn_num else treatment
    perfect_uplift = 2 * (y_true == treatment) + summand

    return uplift_curve(y_true, perfect_uplift, treatment)


def uplift_curve(y_true, uplift, treatment):

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0   # control score
    y_true_trmnt[treatment == 0] = 0  # treatment score

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]


    num_trmnt = stable_cumsum(treatment)#[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)#[threshold_indices]

    num_all = np.array([i for i in range(len(treatment))])

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)#[threshold_indices]

    # (treatment_score / treatment_num)  - (control_score/control_num)*num_all
    # print( f'y_trmnt:{y_trmnt}, num_trmnt:{num_trmnt}, y_ctrl:{y_ctrl},num_ctrl:{num_ctrl}')
    #,casting='unsafe'
    curve_values = (np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0) -
                    np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)) * num_all

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values

def plot_uplift_curve(y_true, uplift, treatment, random=True, perfect=True):

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)
    ax.plot(x_actual, y_actual, label='Model', color='blue')

    if random:
        x_baseline, y_baseline = x_actual, x_actual * \
            y_actual[-1] / len(y_true)
        ax.plot(x_baseline, y_baseline, label='Random', color='black')
        ax.fill_between(x_actual, y_actual, y_baseline, alpha=0.2, color='b')

    if perfect:
        x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
        ax.plot(x_perfect, y_perfect, label='Perfect', color='Red')

    ax.legend(loc='lower right')
    ax.set_title(
        f'auuc={auuc(y_true, uplift, treatment):.4f}')
    ax.set_xlabel('Number targeted')
    ax.set_ylabel('Gain: treatment - control')

    return ax


if __name__ == '__main__':
    # ms = auuc(np.random.randint(2, size=(1000, )), np.random.randn(1000,), np.random.randint(2, size=(1000,)))
    ms = auuc(np.random.randn(1000,), np.random.randn(1000,), np.random.randint(2, size=(1000,)))
    print(ms)
