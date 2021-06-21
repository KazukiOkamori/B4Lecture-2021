import argparse
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# pickle のロード
#   fname: ファイル名
def load_pickle(fname):
    data = pickle.load(open(fname, "rb"))
    answer_models = np.array(data["answer_models"])  # 定義済みHMM
    output = np.array(data["output"])  # 出力系列
    pi = np.squeeze(np.array(data["models"]["PI"]))  # 初期確率
    a = np.array(data["models"]["A"])  # 状態遷移確率行列
    b = np.array(data["models"]["B"])  # 出力確率

    return answer_models, output, pi, a, b


# forward アルゴリズム
#   output: 出力系列
#   pi: 初期確率
#   a: 状態遷移確率行列
#   b: 出力確率
def forward_algorithm(output, pi, a, b):
    n_data, n_series = output.shape  # n_data: データ数, n_series: 出力系列数
    models = np.empty(n_data)  # モデル

    for i in range(n_data):
        alpha = pi * b[:, :, output[i, 0]]  # 初期値

        for j in range(1, n_series):
            alpha = np.sum(a.T * alpha.T, axis=1).T \
                * b[:, :, output[i, j]]  # forward アルゴリズムは合計値

        models[i] = np.argmax(np.sum(alpha, axis=1))  # 同じく合計値

    return models


# viterbi アルゴリズム
#   output: 出力系列
#   pi: 初期確率
#   a: 状態遷移確率行列
#   b: 出力確率
def viterbi_algorithm(output, pi, a, b):
    n_data, n_series = output.shape  # n_data: データ数, n_series: 出力系列数
    models = np.empty(n_data)  # モデル

    for i in range(n_data):
        alpha = pi * b[:, :, output[i, 0]]  # 初期値

        for j in range(1, n_series):
            alpha = np.max(a.T * alpha.T, axis=1).T \
                * b[:, :, output[i, j]]  # viterbi アルゴリズムは最大値

        models[i] = np.argmax(np.max(alpha, axis=1))  # 同じく最大値

    return models


# ヒートマップのプロット
#   models: アルゴリズムから予測したモデル
#   answer_models: 正解ラベルのモデル
#   ax: グラフ領域
#   title: グラフのタイトル
#   time: アルゴリズムの実行時間
def plot_heatmap(models, answer_models, ax, title, time):
    acc = np.sum(models == answer_models) / models.shape[0] * 100  # 精度

    cmx = confusion_matrix(answer_models, models)  # 混合行列
    sns.heatmap(cmx, cmap="binary", annot=True,
                fmt="d", cbar=False, square=True, ax=ax)  # ヒートマップの作成
    ax.set_title(f"{title}\nAcc.{acc} time:{time:.4f}")  # タイトル
    ax.set_xlabel("Predict")  # x軸ラベル
    ax.set_ylabel("Answer")  # y軸ラベル


# main
def main(args):
    fname = args.fname  # 入力ファイル(pickle)
    answer_models, output, pi, a, b = load_pickle(fname)  # pickle データのロード
    fig = plt.figure(figsize=(10.0, 5.0))
    fig.suptitle(f"Heat map ({fname})")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    t_forward_start = time.time()  # forward アルゴリズムが開始
    models_forward = forward_algorithm(output, pi, a, b)  # forward アルゴリズムで計算
    t_forward_to_viterbi = time.time()  # forward 終了 viterbi 開始
    models_viterbi = viterbi_algorithm(output, pi, a, b)  # viterbi アルゴリズムで計算
    t_viterbi_stop = time.time()  # viterbi アルゴリズムが終了

    t_forward = t_forward_to_viterbi - t_forward_start  # forward アルゴリズムの計算時間
    t_viterbi = t_viterbi_stop - t_forward_to_viterbi  # viterbi アルゴリズムの計算時間

    plot_heatmap(models_forward, answer_models, ax1,
                 "Forward algorithm", t_forward)  # forward モデルのヒートマップ
    plot_heatmap(models_viterbi, answer_models, ax2,
                 "Viterbi algorithm", t_viterbi)  # viterbi モデルのヒートマップ

    plt.savefig(f"ex08({fname}).png")  # 保存


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM の分析")
    parser.add_argument("fname", type=str, help="ファイル名")
    args = parser.parse_args()

    main(args)
