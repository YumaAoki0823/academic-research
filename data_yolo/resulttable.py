import pandas as pd
import ast
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定項目
# ==========================================
# ★【重要】headの表を作りたい場合は 'head'、bodyの表を作りたい場合は 'body' に書き換えてください
DATA_TYPES = ['head', 'body']

for DATA_TYPE in DATA_TYPES:
    # 各モデルの名前と、対応するCSVファイルのパスを設定します
    model_files = {
        'yolo26n': f'yolo26n_{DATA_TYPE}.csv',
        'yolo26s': f'yolo26s_{DATA_TYPE}.csv',
        'yolo26m': f'yolo26m_{DATA_TYPE}.csv',
        'yolo26l': f'yolo26l_{DATA_TYPE}.csv',
        'yolo26x': f'yolo26x_{DATA_TYPE}.csv'
    }

    # 抽出条件を設定（ファイルパス内の「停止」という文字列をチェックします）
    TARGET_STATUS_JP = '停止' 

    # パス名に含まれる方位キーワードを検知するためのリスト
    orientation_order = ['Front', 'Behind', 'Left', 'Right', 'None']

    # ==========================================
    # 2. データ集計処理を行う関数
    # ==========================================
    def process_model_csv(file_path):
        # ファイルが存在しない場合は処理をスキップ
        if not os.path.exists(file_path):
            return None
            
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        # 元のCSVの登場順（物理的な行の順番）を記憶するための一時的なIDを付与
        df['Original_Order_Flag'] = range(len(df))
        
        # 「向きの回数」という文字が含まれる列名を自動的に取得
        count_col = next((col for col in df.columns if '向きの回数' in col), None)
        if not count_col:
            print(f"Error: Could not find count column in {file_path}")
            return None
        
        # ファイルパスから「状態（移動/停止）」の部分を抽出
        df['Status_JP'] = df['ファイルパス'].apply(lambda x: x.split('/')[1] if len(x.split('/')) > 1 else '')
        
        # 「停止」データのみにフィルタリング
        df_filtered = df[df['Status_JP'] == TARGET_STATUS_JP].copy()
        
        # 純粋にファイルパスが完全に重複している行（全く同じデータ）のみを除去し、インデックス順を保持
        df_filtered = df_filtered.drop_duplicates(subset=['ファイルパス'], keep='first')
        
        formatted_results = []
        
        # 1行ずつループ処理
        for idx, row in df_filtered.iterrows():
            path = row['ファイルパス']
            parts = path.split('/')
            found_target = next((ori for ori in orientation_order if any(ori.lower() in p.lower() for p in parts)), None)
            
            try:
                # 辞書型の文字列を解析
                d = ast.literal_eval(row[count_col])
                
                # 辞書内から「フレーム数が一番多い1人」のデータだけを厳選
                max_person_counts = [0] * 5  
                max_person_total = -1        
                
                for person_id, counts in d.items():
                    person_total = sum(counts)  
                    if person_total > max_person_total:
                        max_person_total = person_total
                        max_person_counts = counts
                
                sum_all = max_person_total
                
                if sum_all > 0 and found_target:
                    ori_idx = orientation_order.index(found_target)
                    target_count = max_person_counts[ori_idx]
                    pct = round((target_count / sum_all) * 100, 1)
                    display_str = f"{pct}% ({target_count}/{sum_all})"
                else:
                    display_str = "0.0% (0/0)"
            except:
                display_str = "0.0% (0/0)"
                
            formatted_results.append(display_str)
            
        df_filtered['Model_Result'] = formatted_results
        
        # あとでモデル間で紐付けるための共通キー
        df_filtered['Base_Path'] = df_filtered['ファイルパス'].apply(lambda x: '/'.join(x.split('/')[2:6]))
        
        return df_filtered[['Base_Path', 'ファイルパス', 'Model_Result', 'Original_Order_Flag']]

    # ==========================================
    # 3. メイン処理（データの結合と英語への置換）
    # ==========================================
    final_df = None

    for model_name, file_path in model_files.items():
        res = process_model_csv(file_path)
        if res is not None:
            if final_df is None:
                # 表のベース構造を作成
                parts_df = res['ファイルパス'].apply(lambda x: pd.Series({
                    'Camera Position': 'On-board' if '車載' in x.split('/')[2] else 'Roadside',
                    'Distance': x.split('/')[3].replace('距離', ''), 
                    'Target Dir': x.split('/')[4],                  
                    'Trial': x.split('/')[5]                         
                }))
                
                final_df = pd.concat([parts_df, res[['Base_Path', 'Original_Order_Flag']]], axis=1)
                
            # 各モデルの結果データを横並びにマージ
            res_model = res[['Base_Path', 'Model_Result']].rename(columns={'Model_Result': f'{model_name}'})
            final_df = pd.merge(final_df, res_model, on='Base_Path', how='left')

    # ==========================================
    # 4. 並び替えと表画像の生成（PNG出力）
    # ==========================================
    if final_df is not None:
        # 結合用の中間列を削除し、データがない箇所は補完
        final_df = final_df.fillna("0.0% (0/0)")
        
        # 距離を数値（5, 10, 15など）に変換するソート用の一時列を作成
        final_df['Dist_Sort'] = final_df['Distance'].str.replace('m', '', case=False)
        final_df['Dist_Sort'] = pd.to_numeric(final_df['Dist_Sort'], errors='coerce').fillna(999)
        
        # 回数文字列でのソートを廃止！
        # カメラ位置、距離（数値）、方位で並び替えた後、最後の微調整はCSVに登場した「元の物理的な行の順番（Original_Order_Flag）」でソートする
        final_df = final_df.sort_values(by=['Camera Position', 'Dist_Sort', 'Target Dir', 'Original_Order_Flag']).reset_index(drop=True)
        
        # ソート用の一時列を削除
        final_df = final_df.drop(columns=['Base_Path', 'Dist_Sort', 'Original_Order_Flag'])
        
        # 表に表示する文字を綺麗にする（CSVに登場した順を崩さず、上から順に1st, 2nd, 3rdを強制的に割り当て直す、あるいは表記置換）
        final_df['Trial'] = final_df['Trial'].str.replace('1回目', '1st').str.replace('2回目', '2nd').str.replace('3回目', '3rd')
        
        # 列の順番を整理（Camera Position, Distance, Target Dir, Trial の順）
        cols = ['Camera Position', 'Distance', 'Target Dir', 'Trial'] + [col for col in final_df.columns if col not in ['Camera Position', 'Distance', 'Target Dir', 'Trial']]
        final_df = final_df[cols]

        # Matplotlibの描画領域を設定
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        ax.axis('off')
        ax.axis('tight')
        
        # 表オブジェクトを生成
        table = ax.table(cellText=final_df.values, colLabels=final_df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.0)  
        
        # デザイン調整（ヘッダーをダークネイビーに、データ行を交互にグレーに色付け）
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # ヘッダー行
                cell.set_text_props(color='white', weight='bold')
                cell.set_facecolor('#2C3E50') 
            else:  # データ行
                if key[0] % 2 == 0:
                    cell.set_facecolor('#F8F9F9') 
                    
        # 表を保存
        output_filename = f'{DATA_TYPE}_resulttable.png'
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        print(f"エラー: {DATA_TYPE} に対応する有効なCSVファイルが見つかりませんでした。")