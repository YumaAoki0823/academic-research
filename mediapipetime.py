import os
import cv2
import math
import time
import numpy as np
import mediapipe as mp


# 初期設定
FPS = 30

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()   #姿勢推定モデルを読み込む
mp_drawing = mp.solutions.drawing_utils #ランドマーク(骨格点)を描画するユーティリティ


# 動画入力
# フォルダに入れる動画は１つまで（複数入れていると出力した動画が置き換えられるため）
# 動画ファイルのみ抽出
VIDEO_DIR = "data/停止/車載/距離5m/RightandCheck/1回目"
VIDEO_EXTENSIONS = (".mov", ".mp4")
video_files = []
# ビデオファイルのみを抽出
for root, dirs, files in os.walk(VIDEO_DIR):
    for f in files:
        if not f.lower().endswith(VIDEO_EXTENSIONS):
            continue
        if f.startswith("._"):
            continue
        if "output" in f:
            continue
        if "yolo" in f:
            continue
        if f.lower().endswith(VIDEO_EXTENSIONS):
            video_files.append(os.path.join(root, f))

# 抽出された動画ファイルすべてにMediaPipeを適用
for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"can't open: {video_path}")
        continue

    # 出力動画設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        #検出結果付きの映像をMP4ファイルに保存
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_dir = os.path.dirname(video_path)
    output_path = os.path.join(video_dir,"yolo.mov")
    
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

    # 時間計測
    start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:break

        #姿勢推定
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #OpenCVはBGR、MediapipeはRGBであるので変換
        
        results = pose.process(frame_rgb)   #骨格推定
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

        # ---動画出力・表示---------
        out.write(frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    

    cap.release()
    #out.release()
    #cv2.destroyAllWindows()

    # 時間計測終了
    end = time.perf_counter()

    print("処理時間：", end - start, "秒")