import os
import cv2
import csv
import math
import time
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from ultralytics import YOLO


# 定数定義
WIDTH = 1280
HEIGHT = 720
WINDOW_SEC = 0.4


# YOLOのモデル選択
# YOLO26のあとに続くアルファベットで画像の処理精度(n<s<m<l<x)を変更可能。
models = [YOLO("yolo26n-pose.pt"), YOLO("yolo26s-pose.pt"), YOLO("yolo26m-pose.pt"), YOLO("yolo26l-pose.pt"), YOLO("yolo26x-pose.pt")]
model_names = ["yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose"]
movs = [f"{name.split('-')[0]}.mov" for name in model_names]
csvs_video = [f"{name.split('-')[0]}_video.csv" for name in model_names]
csvs_head = [f"{name.split('-')[0]}_head.csv" for name in model_names]
csvs_body = [f"{name.split('-')[0]}_body.csv" for name in model_names]


# 関数定義
# 角度計算
def angle_direction(fps, left, right, direction_count, angle_list):
    # 信頼度が10%以下のときはデータ廃棄
    if left[2] < 0.1 or right[2] < 0.1:
        angle = None
    else:
        # 角度を計算
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        angle = math.atan2(dy, dx)
        angle_list.append(angle)
    
    # 平均フィルターをかける
    # 設定した秒数分のフレーム数を超えた場合リストから要素を削除
    if len(angle_list) > int(fps * WINDOW_SEC):
        angle_list.pop(0)
    
    # Noneを削除
    re_angle_list = [a for a in angle_list if a is not None]

    # 角度がない場合はNoneを返す    
    if len(re_angle_list) == 0:
        smooth_angle = None
        direction = "None"
        direction_count[4] += 1
        return (smooth_angle, direction, direction_count, angle_list)

    # sinとcosに分割して平均を計算
    sin_mean = np.mean(np.sin(re_angle_list))
    cos_mean = np.mean(np.cos(re_angle_list))
    smooth_angle = np.arctan2(sin_mean, cos_mean)
    smooth_angle = np.rad2deg(smooth_angle)

    # 向きを判定
    if -45 <= smooth_angle < 45:
        direction = "Behind"
        direction_count[1] += 1
    elif 45 <= smooth_angle < 135:
        direction = "Right"
        direction_count[3] += 1
    elif -135 <= smooth_angle < -45:
        direction = "Left"
        direction_count[2] += 1
    else:
        direction = "Front"
        direction_count[0] += 1
    
    return (smooth_angle, direction, direction_count, angle_list)

# 少数第4位を四捨五入
def rounding(before):
    after = float(Decimal(str(before)).quantize(Decimal("0.001"), rounding = ROUND_HALF_UP))
    return after


# 定数定義
result_time = []       # モデルごとの1フレームあたりの処理時間のデータ


# メイン処理
for i in range(len(models)):
    # 定数定義
    preprocesstime_per_frame_list = []      # 前処理時間[ms]のリスト
    inferencetime_per_frame_list = []       # 推論の処理時間[ms]のリスト
    postprocesstime_per_frame_list = []     # 後処理時間[ms]のリスト
    calculationtime_per_frame_list = []     # 計算時間[ms]のリスト
    passtime_per_frame_list = []            # 1フレームあたりの処理時間[ms]
    
    # モデル選択
    model = models[i]
    model_name = model_names[i]
    result_video = []
    result_head = []
    result_body = []
    mov = movs[i]
    csv_video = csvs_video[i]
    csv_head = csvs_head[i]
    csv_body = csvs_body[i]


    # 動画入力
    # フォルダに入れる動画は１つまで（複数入れていると出力した動画が置き換えられるため）
    # 動画ファイルのみ抽出
    VIDEO_DIR = "data_yolo"
    VIDEO_EXTENSIONS = (".mov", ".mp4")
    video_files = []
    
    # ビデオファイルのみを抽出
    for root, dirs, files in os.walk(VIDEO_DIR):
        for f in files:
            if not f.lower().endswith(VIDEO_EXTENSIONS) or f.startswith("._") or "yolo" in f:
                continue
            if "resize" in f and f.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, f))


    # 抽出された動画ファイルにYoloposeを適用
    for video_path in video_files:
        
        # 入力動画読み込み
        cap = cv2.VideoCapture(video_path)      # 動画読み込み
        if not cap.isOpened():
            print(f"Can't open the video file:{video_path}")
            continue

        # 出力動画設定
        fps = cap.get(cv2.CAP_PROP_FPS)                                     # 動画のfpsを取得
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        output_path = os.path.join(os.path.dirname(video_path), mov)        # 動画の保存場所指定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')                            # 検出結果付きの映像をMP4ファイルに保存
        out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))    # 出力動画設定


        # 定数定義
        data = {}
        frame_count = 0             # フレーム数
        person_count = 0            # 人の検出回数
        passtime = 0                # 処理時間
        
        person_ids = []

        # 頭と体の向きの回数カウント用
        head_direction_count = {}
        body_direction_count = {}

        # 頭と体の角度計算用
        head_angle_list = {}
        body_angle_list = {}


        # 姿勢推定
        results = model.track(video_path, persist = True, stream = True)

        # フレームごとに逐次処理
        for frame_id, result in enumerate(results):
            frame_count += 1        # フレーム数
            
            # 結果を動画に出力
            annotated = result.plot()


            # 時間計測開始
            start = time.perf_counter()

            
            # 人のidとランドマークを取得
            if result.boxes.id is None:
                ids = []
            else:
                # 検出した人のid取得
                ids = result.boxes.id.cpu().numpy()         # 検出した人のid
                
                # ランドマーク
                kpts_all = result.keypoints.data.cpu().numpy()      # 座標xy、信頼度visibility
                #xyn_all = result.keypoints.xyn.cpu().numpy()        # 正規化座標xy
            data[frame_id] = {}                                     # フレームごとのランドマーク保存用


            # 検出した人ごとに処理
            for idx, person_id in enumerate(ids):
                person_id = int(person_id)
                if not person_id in person_ids:
                    person_ids.append(person_id)

                # 初めて検出した人の定義
                if person_id not in head_direction_count:
                    # 頭と体の向きの回数カウント(Front, Behind, Left, Right, None)
                    head_direction_count[person_id] = [0, 0, 0, 0, 0]
                    body_direction_count[person_id] = [0, 0, 0, 0, 0]

                    # 頭と体の角度計算
                    head_angle_list[person_id] = []
                    body_angle_list[person_id] = []


                # ランドマーク
                kpts = kpts_all[idx]
                #xyn = xyn_all[i]
                data[frame_id][person_id] = kpts

                left_ear = kpts[3]
                right_ear = kpts[4]
                left_shoulder = kpts[5]
                right_shoulder = kpts[6]
                left_hip = kpts[11]
                right_hip = kpts[12]

                # 体の中央座標
                left_body = [(left_shoulder[0] + left_hip[0]) / 2, (left_shoulder[1] + left_hip[1]) / 2, (left_shoulder[2] + left_hip[2]) / 2]
                right_body = [(right_shoulder[0] + right_hip[0]) / 2, (right_shoulder[1] + right_hip[1]) / 2, (right_shoulder[2] + right_hip[2]) / 2]


                # 角度計算
                head_angle = angle_direction(fps, left_ear, right_ear, head_direction_count[person_id], head_angle_list[person_id])
                body_angle = angle_direction(fps, left_body, right_body, body_direction_count[person_id], body_angle_list[person_id])
                angle = [head_angle[0], body_angle[0]]
                direction = [head_angle[1], body_angle[1]]


                # 描画
                offset = idx * 100
                cv2.putText(annotated, f"ID {person_id} Head:{direction[0]}", (50, 50 + offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)      # 頭の向きを画面に表示
                cv2.putText(annotated, f"ID {person_id} Body:{direction[1]}", (50, 100 + offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)     # 体の向きを画面に表示


            # 時間計測終了
            end = time.perf_counter()
            
            # 計測した時間を取得
            preprocess_per_frame = result.speed.get('preprocess', 0.0)          # 前処理（画像サイズ調整など）の処理時間[ms]
            inference_per_frame = result.speed.get('inference', 0.0)            # 推論の処理時間[ms]
            postprocess_per_frame = result.speed.get('postprocess', 0.0)        # 後処理（結果の整理）の処理時間[ms]
            calculationtime_per_frame = (end - start) * 1000                    # 計算（角度など）の処理時間[ms]
            passtime_per_frame = preprocess_per_frame + inference_per_frame + postprocess_per_frame + calculationtime_per_frame     # 1フレームあたりの処理時間[ms]
            passtime += passtime_per_frame / 1000                               # 処理時間の合計[s]

            # 四捨五入
            preprocess_per_frame = rounding(preprocess_per_frame)
            inference_per_frame = rounding(inference_per_frame)
            postprocess_per_frame = rounding(postprocess_per_frame)
            calculationtime_per_frame = rounding(calculationtime_per_frame)
            passtime_per_frame = rounding(passtime_per_frame)

            # 計測した時間を保存
            preprocesstime_per_frame_list.append(preprocess_per_frame)          # 前処理の処理時間
            inferencetime_per_frame_list.append(inference_per_frame)            # 推論の処理時間
            postprocesstime_per_frame_list.append(postprocess_per_frame)        # 後処理の処理時間
            calculationtime_per_frame_list.append(calculationtime_per_frame)    # 計算の処理時間
            passtime_per_frame_list.append(passtime_per_frame)                  # 合計の処理時間
            

            # 結果を保存
            out.write(annotated)

            # 表示
            #cv2.imshow("pose", annotated)
            
            # qを押したら終了
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # 解放
        out.release()
        cv2.destroyAllWindows()

        # 結果整理
        if frame_count >0:
            person_count = len(person_ids)
            length = frame_count / fps      # 動画の長さ[s]
            mean_passtime_per_frame = 1000 * (passtime / frame_count)
            

            # 四捨五入
            length = rounding(length)
            passtime = rounding(passtime)
            mean_passtime_per_frame = rounding(mean_passtime_per_frame)

            # 結果[モデル名、フレーム数、fps[/s]、動画時間[s]、処理時間[s]、1フレームあたりの処理時間[s]、検出した人数、ファイルパス]
            result_video.append([model_name, frame_count, fps, length, passtime, mean_passtime_per_frame, person_count, video_path])
            

            # 顔の方向(Front, Behind, Left, Right, None)のそれぞれの割合
            head_direction_rate = {}

            for person_id in person_ids:
                frame_head_detect = sum(head_direction_count[person_id])
                if frame_head_detect != 0:
                    head_direction_rate[person_id] = np.array(np.array(head_direction_count[person_id]) / frame_head_detect)
                    
                    # 小数点第4位を四捨五入
                    head_direction_rate[person_id] = [float(Decimal(str(num)).quantize(Decimal("0.001"), rounding = ROUND_HALF_UP)) for num in head_direction_rate[person_id]]
            
            # 結果[モデル名、検出した人数、頭の向きの回数(Front, Behind, Left, Right, None)、頭の向きの割合[%]、ファイルパス]
            result_head.append([model_name, person_count, head_direction_count, head_direction_rate, video_path])


            # 体の方向(Front, Behind, Left, Right, None)のそれぞれの割合
            body_direction_rate = {}
            
            for person_id in person_ids:
                frame_body_detect = sum(body_direction_count[person_id])
                if frame_body_detect != 0:
                    body_direction_rate[person_id] = np.array(body_direction_count[person_id]) / frame_body_detect
                    # 小数点第4位を四捨五入
                    body_direction_rate[person_id] = [float(Decimal(str(num)).quantize(Decimal("0.001"), rounding = ROUND_HALF_UP)) for num in body_direction_rate[person_id]]
            
            # 結果[モデル名、検出した人数、体の向きの回数(Front, Behind, Left, Right, None)、体の向きの割合[%]、ファイルパス]
            result_body.append([model_name, person_count, body_direction_count, body_direction_rate, video_path])



    # 結果をcsvに保存
    # 結果の動画情報をcsvに保存
    csv_path = os.path.join(VIDEO_DIR, csv_video)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["モデル名", "フレーム数", "FPS[/s]", "動画時間[s]", "処理時間[s]", "1フレームあたりの処理時間[ms]", "検出した人数", "ファイルパス"])

        writer.writerows(result_video)
    
    # 結果の人の頭の情報をcsvに保存
    csv_path = os.path.join(VIDEO_DIR, csv_head)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["モデル名", "検出した人数", "向きの回数(Front, Behind, Left, Right, None)", "向きの割合[%]", "ファイルパス"])

        writer.writerows(result_head)

    # 結果の人の体の情報をcsvに保存
    csv_path = os.path.join(VIDEO_DIR, csv_body)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["モデル名", "検出した人数", "向きの回数(Front, Behind, Left, Right, None)", "向きの割合[%]", "ファイルパス"])

        writer.writerows(result_body)



    # 処理時間の結果を整理
    time_names = ["prepocesstime（前処理段階）", "inferencetime（推論段階）", "postprocesstime（後処理段階）", "calculationtime（計算段階）", "passtime（合計）"]
    time_lists = [preprocesstime_per_frame_list, inferencetime_per_frame_list, postprocesstime_per_frame_list, calculationtime_per_frame_list, passtime_per_frame_list]

    for j in range(len(time_lists)):
        name = time_names[j]
        list = time_lists[j]
        
        # 処理時間の結果
        arr = np.array(list)
        mean_time = np.mean(arr)     # 1フレームあたりの処理時間の平均値[ms]
        var_time = np.var(arr)       # 1フレームあたりの処理時間の分散値[ms]
        max_time = np.max(arr)       # 1フレームあたりの処理時間の最大値[ms]
        min_time = np.min(arr)       # 1フレームあたりの処理時間の最小値[ms]

        # 四捨五入
        mean_time = rounding(mean_time)
        var_time = rounding(var_time)
        max_time = rounding(max_time)
        min_time = rounding(min_time)

        result_time.append([model_name, name, mean_time, var_time, max_time, min_time])


# 処理時間の結果情報をcsvに保存
csv_path = os.path.join(VIDEO_DIR, "time.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["モデル名", "処理段階", "平均値[ms]", "分散値[ms]", "最大値[ms]", "最小値[ms]"])

    writer.writerows(result_time)