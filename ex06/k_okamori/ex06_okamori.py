import numpy as np
import matplotlib.pyplot as plt
import argparse

# 主成分分析を実装するクラス


class PCA:

    # コンストラクタ
    def __init__(self, fname, data):
        self.fname = fname
        self.data = data  # データ
        self.dim = data.shape[1]  # 次元

    # 標準化(平均: 0, 分散: 1)
    def standardization(self, data):
        x_avg = np.average(data, axis=0)  # 平均
        x_sq_avg = np.average(np.square(data), axis=0)  # 二乗平均
        x_avg_sq = np.square(x_avg)  # 平均の二乗
        x_s = x_sq_avg - x_avg_sq  # 分散 = 二乗平均 - 平均の二乗

        return (data - x_avg) / np.sqrt(x_s)

    # 分散共分散行列の作成
    def var_cov(self, data):
        x_avg = np.average(data, axis=0)  # 平均
        x_avg_pro = np.dot(x_avg.T, x_avg)  # 平均の積
        x_pro_avg = np.dot(data.T, data) / data.shape[0]  # 積の平均

        return x_pro_avg - x_avg_pro  # 共分散 = 積の平均 - 平均の積

    # 主成分分析の計算
    def calculate_pca(self):
        self.data_std = self.standardization(self.data)  # 標準化
        data_var_cov = self.var_cov(self.data_std)  # 分散共分散行列の作成
        eig, v = np.linalg.eig(data_var_cov)  # 固有値を求める
        self.eig_v = np.vstack([eig, v])  # 固有値と固有ベクトルをまとめる
        self.eig_v = self.eig_v[:, np.argsort(-self.eig_v[0])]  # 固有値の降順ソート

    # 主成分のプロット(2次元)
    def plot_pc_2d(self, ax):
        ax.scatter(self.data_std[:, 0], self.data_std[:, 1])
        ax.plot([0, self.eig_v[1, 0]], [0, self.eig_v[2, 0]],
                label=f"z0: ccr = {self.eig_v[0, 0] / self.dim}")
        ax.plot([0, self.eig_v[1, 1]], [0, self.eig_v[2, 1]],
                label=f"z1: ccr = {self.eig_v[0, 1] / self.dim}")
        ax.set_title(f"proncipal component({self.fname})")
        ax.legend()  # レイアウト調整

    # 主成分のプロット(3次元)
    def plot_pc_3d(self, ax):
        ax.scatter(self.data_std[:, 0],
                   self.data_std[:, 1], self.data_std[:, 2])
        ax.plot([0, self.eig_v[1, 0]], [0, self.eig_v[2, 0]],
                [0, self.eig_v[3, 0]], label=f"z0: ccr = {self.eig_v[0, 0] / self.dim}")
        ax.plot([0, self.eig_v[1, 1]], [0, self.eig_v[2, 1]],
                [0, self.eig_v[3, 1]], label=f"z1: ccr = {self.eig_v[0, 1] / self.dim}")
        ax.plot([0, self.eig_v[1, 2]], [0, self.eig_v[2, 2]],
                [0, self.eig_v[3, 2]], label=f"z2: ccr = {self.eig_v[0, 2] / self.dim}")
        ax.set_title(f"proncipal component({self.fname})")
        ax.legend()  # レイアウト調整

    # 次元圧縮のプロット
    def plot_dr(self, ax):
        data_dr = np.dot(self.data_std, self.eig_v[1:])
        ax.scatter(data_dr[:, 0], data_dr[:, 1])
        ax.set_title(f"dimensionality reducion({self.fname})")

    # 累積寄与率のプロット
    def plot_ccr(self, ax):
        ccr = np.cumsum(self.eig_v[0]) / self.dim
        ax.plot(
            np.arange(1, self.eig_v.shape[1] + 1), ccr)
        idx = np.count_nonzero(ccr < 0.9)
        ax.plot(idx, ccr[idx], "ms", ms=10)
        ax.text(idx, ccr[idx]-0.05, f"({idx}, {ccr[idx]})")


# main


def main(args):
    csv = args.fname  # ファイル名

    n_ax = args.pc + args.dr + args.ccr  # オプションの個数
    fig = plt.figure(figsize=(8.0, 6.0 * n_ax))  # グラフ領域の確保

    data = np.loadtxt(fname=csv, dtype="float",
                      delimiter=',')  # csv の読み込み

    pca = PCA(csv, data)  # インスタンス化
    pca.calculate_pca()  # 主成分分析

    ax_num = 0

    if args.pc:  # 主成分のプロット
        if pca.dim == 2:  # 2次元
            ax_num += 1
            ax_pc = fig.add_subplot(n_ax, 1, ax_num)
            pca.plot_pc_2d(ax_pc)

        elif pca.dim == 3:  # 3次元
            ax_num += 1
            ax_pc = fig.add_subplot(n_ax, 1, ax_num, projection="3d")
            pca.plot_pc_3d(ax_pc)

        else:
            print("主成分のプロットは2次元または3次元のみです")

    if args.dr:  # 次元圧縮のプロット
        if pca.dim == 3:
            ax_num += 1
            ax_dr = fig.add_subplot(n_ax, 1, ax_num)
            pca.plot_dr(ax_dr)

        else:
            print("次元圧縮のプロットは3次元のみです")

    if args.ccr:  # 累積寄与率のプロット
        ax_num += 1
        ax_ccr = fig.add_subplot(n_ax, 1, ax_num)
        pca.plot_ccr(ax_ccr)

    fig.tight_layout()  # レイアウト調整
    plt.savefig(f"ex06({csv}).png")

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="主成分分析")
    parser.add_argument("fname", type=str, help="ファイル名")
    parser.add_argument("-pc", action="store_true", help="主成分のプロット")
    parser.add_argument("-ccr", action="store_true", help="寄与率のプロット")
    parser.add_argument("-dr", action="store_true", help="次元圧縮のプロット")
    args = parser.parse_args()

    main(args)
