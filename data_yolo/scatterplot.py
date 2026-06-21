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

    # データ抽出
    # x = 動画の長さ[s]
    # y = 処理時間[s]
    flame = df["動画時間[s]"]
    elapsed = df["処理時間[s]"]

    # 散布図の傾きを算出
    slope, intercept = np.polyfit(flame, elapsed, 1)

    # 散布図とラベルを描画
    plt.scatter(flame, elapsed, label=f"{model} (y={slope:.2f}x)", alpha=0.7)
    
    # 傾きをもとにx = 0~25[s]の範囲で回帰直線を生成
    x_line = np.linspace(0, 25, 100)
    y_line = slope * x_line + intercept
    
    # 回帰直線を描画
    plt.plot(x_line, y_line, linestyle="--", alpha=0.6)


# グラフの設定
plt.title("Video duration and Processing Time")
plt.xlabel("Video duration[s]")
plt.ylabel("Processing time[s]")
plt.xlim(0,25)

# 凡例とグリッド
plt.legend(loc="upper left") # 凡例がデータと被る場合は位置を調整してください
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig("scatterplot.png", bbox_inches="tight")
plt.show()