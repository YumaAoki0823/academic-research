import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# モデルごとのCSVパス
files = {
    "yolo26n" : "yolo26n_video.csv",
    "yolo26s" : "yolo26s_video.csv",
    "yolo26m" : "yolo26m_video.csv",
    "yolo26l" : "yolo26l_video.csv",
    "yolo26x" : "yolo26x_video.csv",
}

data = []
labels = []

for model, path in files.items():
    df = pd.read_csv(path)

    # 経過時間だけ取り出し
    elapsed = df["処理時間[s]"]
    mean = df["処理時間[s]"].mean()
    var = df["処理時間[s]"].var()
    min = df["処理時間[s]"].min()
    max = df["処理時間[s]"].max()
    print("モデル：", model, "平均：", f"{mean:.3f}", "分散：", f"{var:.3f}", "最小：", f"{min:.3f}", "最大：", f"{max:.3f}", "\n")

    data.append(elapsed)
    labels.append(model)

# 箱ひげ図
plt.figure()
plt.boxplot(data, labels=labels, showfliers=False)

plt.title("Elapsed time by model")
plt.xlabel("Model")
plt.ylabel("Time[s]")
plt.savefig("boxplot.png")
plt.show()