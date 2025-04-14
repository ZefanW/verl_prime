# calculate AUC from the csv
# for results matched by given regex pattern, do bayesian optimization, show landscape, and give next recommandation

import pandas as pd
import numpy as np
import re
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import json

from matplotlib import gridspec, pyplot as plt


def ema(arr, factor=0.9):
    """
    计算 numpy 1D array 的指数移动平均（EMA）。

    参数：
    - arr: 一维 numpy 数组
    - factor: 平滑因子，取值范围 [0,1]，越大则近期数据权重越高，默认为 0.9

    返回：
    - 一个一维 numpy 数组，为原数组的 EMA 序列
    """
    if len(arr) == 0:
        return np.array([])

    ema_arr = np.empty_like(arr, dtype=float)
    ema_arr[0] = arr[0]  # 初始化第一个值
    # 计算 EMA，每一步按公式：EMA[i] = factor * arr[i] + (1 - factor) * EMA[i-1]
    for i in range(1, len(arr)):
        ema_arr[i] = factor * arr[i] + (1 - factor) * ema_arr[i - 1]
    return ema_arr
def calc_auc(csv_file_name, total_steps):
    csv_file=pd.read_csv(csv_file_name)

    score_dict={}

    for k in csv_file.columns:
        score_list=csv_file[k].to_numpy()
        score_list=score_list[~np.isnan(score_list)]
        score_list=ema(score_list)
        if len(score_list)<total_steps:
            score_list=np.pad(score_list, (0, total_steps-len(score_list)), mode='constant', constant_values=0)
        else:
            score_list=score_list[:total_steps]
        score_list=np.maximum.accumulate(score_list)

        auc=score_list.sum()/total_steps
        score_dict[k]=float(auc)

    print(json.dumps(score_dict, indent=4))

    return score_dict

def posterior(optimizer, grid):
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y=None):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontsize=30
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    optimizer.acquisition_function._fit_gp(optimizer._gp, optimizer._space)
    mu, sigma = posterior(optimizer, x)

    # axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((0, 1))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})


    utility_function = acquisition.UpperConfidenceBound(kappa=2.5)
    utility = -1 * utility_function._get_acq(gp=optimizer._gp)(x)
    x = x.flatten()

    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0,1))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
def bayesian_optimize(score_dict, pattern):
    re_pattern = re.compile(pattern)
    lam2score={}
    for k,v in score_dict.items():
        match = re_pattern.match(k)
        if match:
            lam2score[float(match.group(1))]=v
    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acquisition.UpperConfidenceBound(kappa=2.5),
        pbounds={'x': (0, 1.0)},
        verbose=2,
        random_state=1,
    )
    for k,v in lam2score.items():
        optimizer.register(params={'x':k}, target=v)
    optimizer.register(params={'x':1.0}, target=0.85)
    next_point_to_probe=optimizer.suggest()
    print('next lambda to probe: '+str(next_point_to_probe))
    plot_gp(optimizer, np.linspace(0,1.0,100).reshape(-1,1))
    plt.show()



if __name__=='__main__':
    score_dict = calc_auc('/home/wangzefan/data/verl_prime/eval_results/wandb_export_2025-04-14T22_41_41.917+08_00.csv',255)

    bayesian_optimize(score_dict, r"^prime-([0-9]+\.[0-9]+)-strict-dpo-tll-freeze-lowbeta-clip-clipvalue - acc$")