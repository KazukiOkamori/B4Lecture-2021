import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import argparse


# csv ファイルのロード
def load_file(csv):
    data = np.loadtxt(fname=csv, dtype="float",
                      delimiter=',')  # csv の読み込み
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    dim = int(data.shape[1])  # dim: データの次元

    return data, dim


# パラメータの初期化
def init_param(dim, n_cluster):
    pi = np.full(n_cluster, 1 / n_cluster)  # pi: 混合係数
    mu = np.random.randn(n_cluster, dim)  # mu: 平均
    sigma = np.array([np.eye(dim) for i in range(n_cluster)])  # sigma: 分散

    return pi, mu, sigma


# ガウス分布の計算
#   data: 分析するデータ
#   dim: データの次元
#   mu: 平均
#   sigma: 分散
def calc_gaussian(data, dim,  mu, sigma):
    return (np.exp(-1 / 2 * np.diag((data - mu) @ np.linalg.inv(sigma) @ (data-mu).T))
            / np.sqrt(((2 * np.pi) ** dim) * np.linalg.det(sigma)))


# 混合ガウス分布の計算
#   data: 分析するデータ
#   dim: データの次元
#   n_cluster: クラスター数
#   pi: 混合係数
#   mu: 平均
#   sigma: 分散
def calc_mixed_gaussian(data, dim, n_cluster, pi, mu, sigma):
    gaussian = np.zeros((data.shape[0], n_cluster))

    for i in range(n_cluster):  # 混合係数にガウス分布を掛けて和をとる
        gaussian[:, i] = pi[i] * calc_gaussian(data, dim, mu[i], sigma[i])

    mixed_gaussian = np.sum(gaussian, axis=1).reshape(-1, 1)

    return gaussian, mixed_gaussian


# 対数尤度の計算
#   mixed_gaussian: 混合ガウス分布
def calc_log_likelihood(mixed_gaussian):
    return np.sum(np.log(mixed_gaussian))


# EMアルゴリズム
#   data: 分析するデータ
#   dim: データの次元
#   n_cluster: クラスター数
#   pi: 混合係数
#   mu: 平均
#   sigma: 分散
#   gap: 対数尤度の収束基準
def em_algorithm(data, dim, n_cluster, pi, mu, sigma, gap):

    n_data = data.shape[0]  # n_data: データ数
    gaussian, mixed_gaussian = calc_mixed_gaussian(
        data, dim, n_cluster, pi, mu, sigma)

    likelihood_previous = -np.inf  # likelihood_previous: 前回の対数尤度
    likelihood = calc_log_likelihood(mixed_gaussian)
    likelihood_list = np.array([likelihood])  # 対数尤度の推移を記録

    while likelihood - likelihood_previous > gap:  # 収束基準を満たすまでループ
        gamma = gaussian / mixed_gaussian  # gamma: 負担率

        n_k = np.sum(gamma, axis=0)

        pi = n_k / n_data

        mu = gamma.T @ data / n_k.reshape(-1, 1)

        for i in range(n_cluster):
            sigma[i] = gamma[:, i] * (data - mu[i]).T @ (data - mu[i]) / n_k[i]

        gaussian, mixed_gaussian = calc_mixed_gaussian(
            data, dim, n_cluster, pi, mu, sigma)

        likelihood_previous = likelihood
        likelihood = calc_log_likelihood(mixed_gaussian)
        likelihood_list = np.append(likelihood_list, likelihood)

    return pi, mu, sigma, likelihood_list


# 対数尤度のプロット
#   likelihood_list: 対数尤度のリスト
#   ax: グラフ領域
def plot_likelihood_list(likelihood_list, ax):
    ax.plot(likelihood_list)
    ax.set_xlabel("iter")
    ax.set_ylabel("log_likelihood")


# 散布図のプロット
#   data: 分析するデータ
#   n_cluster: クラスター数
#   pi: 混合係数
#   mu: 平均
#   sigma: 分散
#   ax: グラフ領域
def plot_scatter(data, n_cluster, pi, mu, sigma, ax):

    ax.scatter(data, np.zeros(data.shape[0]), c="blue")  # 散布図
    ax.scatter(mu, np.zeros(mu.shape[0]), c="red")  # クラスタ
    # 以下，正規分布
    gaussian_x = np.linspace(data.min(), data.max(), 100)
    gaussian_y = np.zeros((100, n_cluster))

    for i in range(n_cluster):
        gaussian_y[:, i] = pi[i] * \
            multivariate_normal.pdf(gaussian_x, mu[i], sigma[i])
        ax.plot(gaussian_x, gaussian_y[:, i])

    ax.plot(gaussian_x, np.sum(gaussian_y, axis=1))


# 散布図のプロット(2次元)
#   data: 分析するデータ
#   n_cluster: クラスター数
#   pi: 混合係数
#   mu: 平均
#   sigma: 分散
#   ax: グラフ領域
def plot_scatter_2d(data, n_cluster, pi, mu, sigma, ax):
    ax.scatter(data[:, 0], data[:, 1], c="blue")  # 散布図
    ax.scatter(mu[:, 0], mu[:, 1], c="red")  # クラスタ
    # 以下，等高線
    x = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 0.01)
    y = np.arange(np.min(data[:, 1]), np.max(data[:, 1]), 0.01)
    x, y = np.meshgrid(x, y)
    z = np.squeeze(np.array([calc_mixed_gaussian(i, 2, n_cluster, pi, mu, sigma)[1]
                             for i in np.dstack((x, y))]))
    ax.contour(x, y, z)


# main
def main(args):
    data, dim = load_file(args.fname)
    n_cluster = args.n_cluster
    fig = plt.figure(figsize=(6.0, 10.0))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    pi, mu, sigma = init_param(dim, n_cluster)  # パラメータの初期化

    pi, mu, sigma, likelihood_list = \
        em_algorithm(data, dim, n_cluster, pi, mu, sigma, 0.001)  # EMアルゴリズム

    plot_likelihood_list(likelihood_list, ax1)  # 対数尤度のプロット

    if dim == 1:  # 1次元
        plot_scatter(data, n_cluster, pi, mu, sigma, ax2)

    if dim == 2:  # 2次元
        plot_scatter_2d(data, n_cluster, pi, mu, sigma, ax2)

    plt.savefig(f"ex07({args.fname}).png")  # 保存


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMMの分析")
    parser.add_argument("fname", type=str, help="ファイル名")
    parser.add_argument("n_cluster", type=int, help="クラスター数")
    args = parser.parse_args()

    main(args)
