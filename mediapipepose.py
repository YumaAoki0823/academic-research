import os
import cv2
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# ---初期設定---------
FPS = 30.0
THRESHOLD = 0.05            # 移動/停止の閾値
ABNORMAL = 3.0              # 異常値
WINDOW_SEC = 0.8            # 速度平均フィルター
CROSSWALK = "Right"         # 横断歩道の位置（"Front" or "Behind" or "Left" or "Right"）
STREET = ["Front","Behind"] # 横断歩道の位置（"Front" or "Behind" or "Left" or "Right"）
OPPOSIT = "Left"            # 横断歩道の位置（"Front" or "Behind" or "Left" or "Right"）

# カメラ入力(0) or 動画入力(1)
input_style = 1

# ---初期化---------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()   #姿勢推定モデルを読み込む
mp_drawing = mp.solutions.drawing_utils #ランドマーク(骨格点)を描画するユーティリティ


# ---角度と向きを計算---------
def calc_angle_direction(right_landmark,left_landmark):
    # 角度を計算
    dx = right_landmark.x - left_landmark.x
    # dy = right_landmark.y - left_landmark.y
    dz = right_landmark.z - left_landmark.z
    angle = math.degrees(math.atan2(dz, dx))
    return angle


# ---角度ノイズ低減と歩行者の正面角度を計算---------
def angle_noizereduction(rough_angle, angle_list, smooth_angle_list):
    angle_list.append(rough_angle)
    # 0.8秒の平均フィルターをかける
    if len(angle_list) > int(FPS * WINDOW_SEC):
        angle_list.pop(0)
    
    angles_rad = np.deg2rad(angle_list)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    smooth_angle = np.arctan2(sin_mean, cos_mean)
    smooth_angle = np.rad2deg(smooth_angle)
    smooth_angle_list.append(smooth_angle)
    
    # 向きを判定
    if 45 <= smooth_angle <= 135:
        direction = "Left"
    elif -135 <= smooth_angle <= -45:
        direction = "Right"
    elif -45 < smooth_angle < 45:
        direction = "Behind"
    else:
        direction = "Front"

    return (smooth_angle, direction)


# ---前フレームから現フレームの速度を計算---------
def calc_speed(dt, ids, prev_center):
    xs = zs = ws = 0.0
    valid = False
    for id in ids:
        # 信頼度0.5以下を排除
        if id.visibility > 0.5:
            xs += id.x * id.visibility
            zs += id.z * id.visibility
            ws += id.visibility
            valid = True

    if not valid:
        center = None
    else:
        center_x = xs / ws      # 体の中心のx座標
        center_z = zs / ws      # 体の中心のz座標
    center = [center_x, center_z]

    # 速度を計算
    if prev_center == None:
        speed = xspeed = zspeed = 0             # 前フレームの体の中心座標がない場合はnp.nanを使用
    
    elif center == None:
        speed = xspeed = zspeed = np.nan        # 前フレームの体の中心座標がない場合はnp.nanを使用
    
    else:
        dx = center_x - prev_center[0]
        dz = center_z - prev_center[1]
        dist = math.sqrt(dx*dx + dz*dz)
        speed = dist / dt     # 速度
        xspeed = dx / dt      # x軸方向の速度
        zspeed = -dz / dt     # z軸方向の速度
    
    rough_speed = [speed, xspeed, zspeed]
    return (center,rough_speed)     # 数値を求められなかった場合はnp.nanを使用


# ---速度ノイズ低減と歩行者の状態を計算---------
def speed_noizereduction(rough_speed, speed_list, smooth_speed_list):
    speed_list.append(rough_speed)
    # 0.8秒の平均フィルターをかける
    if len(speed_list) > int(FPS * WINDOW_SEC):
        speed_list.pop(0)
    
    # np.nanを弾いたリストのみでの速度を求める
    smooth_speed = np.nanmean(speed_list)
    smooth_speed_list.append(smooth_speed)
    
    if 0 <= smooth_speed <= THRESHOLD:
        state = "Stop"
    elif THRESHOLD <= smooth_speed < ABNORMAL:
        state = "Moving"
    else:
        state = "Abnormal"
    return (smooth_speed, state)

# ---横断意図判定---------
def judge(intention, text, count, head_angle_change_list, direction, speed, state):
    if not intention[0]:
        if state == "Stop":
            intention[0] = True             # 停止している場合、Intention[0]はTrue

    n = len(head_angle_change_list)
    if not intention[1] and n != 2:
        if head_angle_change_list[n-1] >= 80:
            intention[1] = True             # 首の角度の変化率が80度を以上の場合、Intention[1]はTrue

    if intention[2] != "crosswalk":
        for i in range(len(direction)):
            if direction[i] == OPPOSIT:
                intention[2] = "opposit"    # 体の向きが横断歩道の逆側を向いていた場合、Intention[2]は"street"

        for i in range(len(direction)):
            if direction[i] in STREET:
                intention[2] = "street"     # 体の向きが歩道を向いていた場合、Intention[2]は"street"

        # 首での横断意図判定
        if direction[0] == CROSSWALK:
            count[0] += 1
        else:
            count[0] = 0
        # 肩での横断意図判定
        if direction[1] == CROSSWALK:
            count[1] += 1
        else:
            count[1] = 0
        # 腰での横断意図判定
        if direction[2] == CROSSWALK:
            count[2] += 1
        else:
            count[2] = 0
        if any (c >= FPS * 1.0 for c in count):
            intention[2] = "crosswalk"      # 体の向きが1.0秒以上横断歩道を向いていた場合、Intention[2]は"crosswalk"


    if text == "High Potential Intention":
        pass
    elif text == "Potential Intention":
        if intention[2] == "crosswalk":
            text = "High Potential Intention"           # 体が横断歩道側を向いていた場合は横断意図可能性高
        elif intention[2] == "street" and (intention[0] == True or intention[1] == True):
            text = "High Potential Intention"           # 体が歩道側を向いていて、停止しているか首の角度の変化率が秒速80度以上の場合は横断意図可能性高
        else:
            text = "Potential Intention"                # それ以外の場合は横断意図あり 
    else:
        if intention[2] == "crosswalk":
            text = "High Potential Intention"           # 体が横断歩道側を向いていた場合は横断意図可能性高
        elif intention[2] == "street" and (intention[0] == True or intention[1] == True):
            text = "High Potential Intention"           # 体が歩道側を向いていて、停止しているか首の角度の変化率が秒速80度以上の場合は横断意図可能性高
        elif intention[2] == "opposit":
            text = "Low Potential Intention"            # 体が横断歩道と反対側を向いていた場合は横断意図可能性低
        else:
            text = "Potential Intention"                # それ以外の場合は横断意図あり
        
    return (intention, count, text)

# 動画処理関数-
def mediapipe(cap, video_path):
    # 動画情報取得
    with mp_pose.Pose() as pose:
        # FPS,動画幅、動画高さ取得
        dt = 1.0/FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        # 出力動画設定
        if video_path != None:
            video_dir = os.path.dirname(video_path)
            output_path = os.path.join(video_dir,"output.mov")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')        #検出結果付きの映像をMP4ファイルに保存
            out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
        else:
            out = cv2.VideoWriter("output.mov", fourcc, FPS, (width, height))
        

        # ---データ定義---------
        # 首のヨー角計算用データ
        prev_head_angle = 0                                                             # 前フレームの頭の角度
        head_angle_list = shld_angle_list = hip_angle_list = []                         # 平均フィルター用ヨー角リスト
        smooth_head_angle_list = smooth_shld_angle_list = smooth_hip_angle_list = []    # 平滑化ヨー角リスト（グラフ描画で使用）
        head_angle_change_list = []                                                     # 首のヨー角変化率リスト（グラフ描画で使用）
        
        # 速度計算用データ
        prev_center = None                                                              # 前フレームの体の中心座標
        speed_list = xspeed_list = zspeed_list = []                                     # 平均フィルター用速度リスト
        smooth_speed_list = smooth_xspeed_list = smooth_zspeed_list = []                # 平滑化速度リスト（グラフ描画で使用）

        # 横断意図判定用変数
        state_intention = False                                                         # 移動中/停止状態による横断意図フラグ
        head_angle_intention = False                                                    # 頭の向きの変化率によるによる横断意図フラグ
        direction_intention = None                                                      # 向きによる横断意図フラグ
        intention = [state_intention, head_angle_intention, direction_intention]        # 横断意図推定
        head_count = shld_count = hip_count = 0
        count = [head_count, shld_count, hip_count]                                     # 横断歩道を向いていたフレーム数（横断意図判定judgeで使用）
        text = "Low Potential Intention"                                                # 3段階の横断意図判定（「横断可能性高」、「横断可能性あり」、「横断可能性低」）

        # フレーム数カウント
        count_frame = 0
        

        # ---フレームごとの処理---------
        while cap.isOpened():
            ret, frame = cap.read() # カメラから1フレームずつ取得
            count_frame += 1
            if not ret:
                break
        
            # MediaPipeで姿勢推定
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #OpenCVはBGR、MediapipeはRGBであるので変換
            results = pose.process(frame_rgb)   #骨格推定
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark


                # 左右の肩と腰
                left_head = landmarks[7]
                right_head = landmarks[8]
                left_shld = landmarks[11]
                right_shld = landmarks[12]
                left_hip = landmarks[23]
                right_hip = landmarks[24]


                # ---角度計算---------
                # 首・肩・腰の角度[°]を取得
                rough_head_angle = calc_angle_direction(right_head,left_head)       # 現フレームの首のヨー角を計算
                rough_shld_angle = calc_angle_direction(right_shld,left_shld)       # 現フレームの肩のヨー角を計算
                rough_hip_angle = calc_angle_direction(right_hip,left_hip)          # 現フレームの腰のヨー角を計算
                

                # 平均フィルターを用いて平滑化角度[°]と向きをを取得
                (smooth_head_angle, head_direction) = angle_noizereduction(rough_head_angle, head_angle_list, smooth_head_angle_list)       # 首の平滑化ヨー角を計算
                (smooth_shld_angle, shld_direction) = angle_noizereduction(rough_shld_angle, shld_angle_list, smooth_shld_angle_list)       # 肩の平滑化ヨー角を計算
                (smooth_hip_angle, hip_direction) = angle_noizereduction(rough_hip_angle, hip_angle_list, smooth_hip_angle_list)            # 腰の平滑化ヨー角を計算
                

                # 首のヨー角変化率[°/s]を取得
                if prev_head_angle != None:
                    diff = abs(((smooth_head_angle - prev_head_angle) + 180) % 360 -180)       # 現フレームと前フレームの角度の差分を計算
                    if diff > 90:
                        diff = np.nan
                    else:
                        diff *= FPS
                else:
                    diff = 0
                head_angle_change_list.append(diff)     
                prev_head_angle = smooth_head_angle


                # ---速度計算---------
                # 体の中心座標と速度を取得
                ids = [left_shld,right_shld,left_hip,right_hip]             # 肩と腰を用いて中心座標を計算
                (center,rough_speed) = calc_speed(dt, ids, prev_center)     # （体の中心座標、速度）を計算
                if center == None:
                    prev_center = None
                else:
                    prev_center = [center[0],center[1]]


                # 平均フィルターを用いて平滑化速度を取得
                (smooth_speed, state) = speed_noizereduction(rough_speed[0], speed_list, smooth_speed_list)         # (平滑化速度、状態)を計算
                (smooth_xspeed, xstate) = speed_noizereduction(rough_speed[1], xspeed_list, smooth_xspeed_list)     # x軸（画面横）方向の(平滑化速度、状態)を計算
                (smooth_zspeed, zstate) = speed_noizereduction(rough_speed[2], zspeed_list, smooth_zspeed_list)     # z軸（奥行き）方向の(平滑化速度、状態)を計算
                

                # ---横断意図判定---------
                angle = [smooth_head_angle, smooth_shld_angle, smooth_hip_angle]        # 平滑化された各部位の正面角度
                direction = [head_direction, shld_direction, hip_direction]             # 各部位の向き
                speed = [smooth_speed, smooth_xspeed, smooth_zspeed]                    # 平滑化速度
                (intention, count, text) = judge(intention, text, count, head_angle_change_list, direction, speed, state)


                # ---デバッグ出力---------
                print(f"フレーム:{count_frame}\n")
                print(f"首の正面角度とその向き:{angle[0]:.2f}度 {direction[0]}\n")
                print(f"肩の正面角度とその向き:{angle[1]:.2f}度 {direction[1]}\n")
                print(f"腰の正面角度とその向き:{angle[2]:.2f}度 {direction[2]}\n")
                print(f"ピクセル速度:{speed[0]:.4f} (px/s)\n")
                print(f"ピクセルx速度:{speed[1]:.4f} (px/s)\n")
                print(f"ピクセルz速度:{speed[2]:.4f} (px/s)\n")
                print(f"状態:{state}\n")
                print(f"横断意図判定:{intention}\n")


                # ---描画---------
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"Neck:{direction[0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)             # 首の向きを画面に表示
                cv2.putText(frame, f"Shoulder:{direction[1]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)        # 肩の向きを画面に表示
                cv2.putText(frame, f"Hip:{direction[2]}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)             # 腰の向きを画面に表示
                cv2.putText(frame, f"Pixel Speed:{speed[0]:.4f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)     # 速度を画面に表示
                cv2.putText(frame, f"State:{state}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)                  # 歩行者の状態を画面に表示
                cv2.putText(frame, f"Intention To Cross:{text}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)      # 横断意図の有無を画面に表示


            # ---動画出力・表示---------
            out.write(frame)
            cv2.imshow('Pose Detection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        
        # ---角度と速度をグラフ化---------
        # 速度をグラフ化
        time_sec = [i/FPS for i in range(len(smooth_speed_list))]
        smooth_speed_list = [i*FPS for i in smooth_speed_list]
        plt.plot(time_sec, smooth_speed_list, color = "black")
        plt.xlabel("Time[s]")
        plt.xlim(0,None)
        plt.ylabel("Walking speed[px/s]")
        plt.axhline(y = THRESHOLD*FPS, linestyle = "--", color = "blue")
        plt.ylim(0, 10)
        plt.savefig(video_dir + "/walikingspeed.png")
        plt.clf()

        # 角度をグラフ化
        time_sec = [i/FPS for i in range(len(smooth_head_angle_list))]
        plt.plot(time_sec, smooth_head_angle_list, color = "black")
        plt.xlabel("Time[s]")
        plt.xlim(0, None)
        plt.ylabel("Head yaw[°]")
        plt.ylim(-180, 180)
        plt.savefig(video_dir + "/headyaw.png")
        plt.clf()

        # 角度変化率をグラフ化
        time_sec = [i/FPS for i in range(len(head_angle_change_list))]
        plt.plot(time_sec, head_angle_change_list, color = "black")
        plt.xlabel("Time[s]")
        plt.xlim(0, None)
        plt.ylabel("Head yaw change rate[°/s]")
        plt.ylim(0, 140)
        plt.axhline(y = 80, linestyle = "--", color = "blue")
        plt.savefig(video_dir + "/headyawchange.png")
        plt.clf()


        # 終了処理
        out.release()
        cap.release()
        cv2.destroyAllWindows()

# ---動画入力---------
# フォルダに入れる動画は１つまで（複数入れていると出力した動画が置き換えられるため）
if input_style == 1:
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
        
        mediapipe(cap, video_path)
# カメラ入力
elif input_style == 0:
    cap = cv2.VideoCapture(input_style)
    mediapipe(cap, None)