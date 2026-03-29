# -*- coding: utf-8 -*-
"""
データ前処理スクリプト: 山口県橋梁維持管理クラスタリングMVP
- 橋梁データ、人口統計、財政力指数を統合
"""

import pandas as pd
import numpy as np
from datetime import datetime
import config

def load_bridge_data():
    """山口県橋梁データを読み込む（マルチヘッダー対応）"""
    print("🌉 橋梁データを読み込み中...")
    try:
        # マルチヘッダー（0,1行目）を読み込み、1行目の列名を使用
        df = pd.read_excel(config.BRIDGE_DATA_FILE, header=[0, 1])
        
        # 列名を単純化（Unnamed列は0行目の名前を使用）
        df.columns = [col[0] if 'Unnamed' in str(col[1]) else col[0] for col in df.columns]
        
        print(f"  ✓ 橋梁データ: {len(df)}件")
        return df
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return None

def load_fiscal_data():
    """財政力指数データを読み込む（ヘッダーは2行目）"""
    print("💰 財政力指数データを読み込み中...")
    try:
        # 1行目をスキップして2行目をヘッダーとして読み込み
        df = pd.read_excel(config.FISCAL_DATA_FILE, skiprows=1)
        print(f"  ✓ 財政データ: {len(df)}件 (ヘッダー: 2行目)")
        return df
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return None

def load_population_data():
    """人口統計データを読み込む（2行目をヘッダー、最初のデータ行を削除）"""
    print("👥 人口統計データを読み込み中...")
    try:
        # 2行目をヘッダーとして読み込み
        df = pd.read_excel(config.POPULATION_DATA_FILE, header=1)
        
        # 最初の行（元の3行目のヘッダー情報）を削除
        df = df.drop(index=0).reset_index(drop=True)
        
        # Unnamed列に適切な名前を設定
        rename_dict = {
            'Unnamed: 0': '団体コード',
            'Unnamed: 1': '都道府県名',
            'Unnamed: 2': '市区町村名',
            'Unnamed: 3': '性別'
        }
        df = df.rename(columns=rename_dict)
        
        print(f"  ✓ 人口データ: {len(df)}件")
        return df
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return None

def calculate_bridge_age(df, construction_year_col=None):
    """橋齢を計算する"""
    current_year = datetime.now().year
    
    # 架設年の列名を推定（実際のデータに応じて調整）
    possible_cols = ['架設年（西暦）', '架設年次', '建設年', '竣工年', '年次', '架設年度']
    year_col = None
    
    if construction_year_col and construction_year_col in df.columns:
        year_col = construction_year_col
    else:
        for col in possible_cols:
            if col in df.columns:
                year_col = col
                break
    
    if year_col:
        # 「不明」を欠損値として扱い、数値に変換
        df[year_col] = df[year_col].replace('不明', np.nan)
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        # 欠損値（「不明」を含む）を中央値で補完
        median_year = df[year_col].median()
        df[year_col] = df[year_col].fillna(median_year)
        
        # 橋齢を計算
        df['bridge_age'] = current_year - df[year_col]
        df['bridge_age'] = df['bridge_age'].clip(lower=0, upper=150)  # 異常値を除外
        
        print(f"  ✓ 架設年の列「{year_col}」を使用（中央値: {int(median_year)}年）")
    else:
        print("  ⚠ 架設年の列が見つかりません。橋齢をデフォルト値50で設定します。")
        df['bridge_age'] = 50
    
    return df

def extract_condition_score(df):
    """健全度スコアを抽出・数値化する"""
    # 健全度の列（○が入っている列を特定）
    condition_cols = ['健全度Ⅰ', '健全度Ⅱ', '健全度Ⅲ', '健全度Ⅳ']
    
    # すべての健全度列が存在するか確認
    if all(col in df.columns for col in condition_cols):
        print(f"  ✓ 健全度列を検出: {', '.join(condition_cols)}")
        
        # 各行で「○」が入っている列を特定し、スコアを割り当て
        def get_condition_score(row):
            if row['健全度Ⅰ'] == '○':
                return 1
            elif row['健全度Ⅱ'] == '○':
                return 2
            elif row['健全度Ⅲ'] == '○':
                return 3
            elif row['健全度Ⅳ'] == '○':
                return 4
            else:
                return 2  # デフォルト値
        
        df['condition_score'] = df.apply(get_condition_score, axis=1)
        
        # 健全度の分布を表示
        condition_counts = df['condition_score'].value_counts().sort_index()
        print(f"  健全度分布:")
        for score, count in condition_counts.items():
            print(f"    健全度{['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ'][int(score)-1]}: {count}件")
    else:
        print("  ⚠ 健全度の列が見つかりません。デフォルト値2で設定します。")
        df['condition_score'] = 2
    
    return df

def calculate_maintenance_priority(df):
    """補修優先度を計算する（橋齢と健全度から）"""
    df['maintenance_priority'] = df['bridge_age'] * df['condition_score']
    return df

def extract_structure_category(df):
    """橋梁の種類から構造形式のカテゴリーを抽出"""
    print("  🏗️ 構造形式カテゴリーを抽出中...")
    
    if '橋梁の種類' not in df.columns:
        print("    ⚠ 「橋梁の種類」列が見つかりません。デフォルト値0を設定します。")
        df['structure_category'] = 0
        return df
    
    # 構造形式のカテゴリー分類
    def categorize_structure(bridge_type):
        if pd.isna(bridge_type):
            return 0  # 不明
        
        bridge_type_str = str(bridge_type).upper()
        
        # RC系（鉄筋コンクリート）
        if 'RC' in bridge_type_str or 'RCT' in bridge_type_str or 'RC床版' in bridge_type_str:
            return 1
        # PC系（プレストレストコンクリート）
        elif 'PC' in bridge_type_str or 'PCT' in bridge_type_str or 'PC床版' in bridge_type_str:
            return 2
        # 鋼橋（鋼材）
        elif '鋼' in bridge_type_str or 'STEEL' in bridge_type_str:
            return 3
        # ボックス系
        elif 'ボックス' in bridge_type_str or 'BOX' in bridge_type_str:
            return 4
        # その他
        else:
            return 5
    
    df['structure_category'] = df['橋梁の種類'].apply(categorize_structure)
    
    # カテゴリー分布を表示
    category_names = {0: '不明', 1: 'RC系', 2: 'PC系', 3: '鋼橋', 4: 'ボックス系', 5: 'その他'}
    category_counts = df['structure_category'].value_counts().sort_index()
    print(f"    構造形式分布:")
    for cat, count in category_counts.items():
        print(f"      {category_names.get(cat, '未定義')}: {count}件")
    
    return df

def calculate_bridge_area(df):
    """橋面積を計算（橋長×幅員）"""
    print("  📐 橋面積を計算中...")
    
    if '橋長（m）' not in df.columns or '幅員（m）' not in df.columns:
        print("    ⚠ 橋長または幅員の列が見つかりません。デフォルト値100を設定します。")
        df['bridge_area'] = 100.0
        return df
    
    # 数値化（エラーがある場合はNaNに）
    df['橋長（m）'] = pd.to_numeric(df['橋長（m）'], errors='coerce')
    df['幅員（m）'] = pd.to_numeric(df['幅員（m）'], errors='coerce')
    
    # 橋面積を計算
    df['bridge_area'] = df['橋長（m）'] * df['幅員（m）']
    
    # 欠損値や異常値を中央値で補完
    median_area = df['bridge_area'].median()
    df['bridge_area'] = df['bridge_area'].fillna(median_area)
    df['bridge_area'] = df['bridge_area'].clip(lower=1, upper=10000)  # 異常値除外
    
    print(f"    ✓ 橋面積: 平均={df['bridge_area'].mean():.1f}m², 中央値={median_area:.1f}m²")
    
    return df

def extract_emergency_route_dummy(df):
    """緊急輸送道路ダミー変数を抽出（○=1, それ以外=0）"""
    print("  🚨 緊急輸送道路ダミー変数を抽出中...")
    
    if '緊急輸送道路' not in df.columns:
        print("    ⚠ 「緊急輸送道路」列が見つかりません。デフォルト値0を設定します。")
        df['emergency_route'] = 0
        return df
    
    df['emergency_route'] = df['緊急輸送道路'].apply(lambda x: 1 if x == '○' else 0)
    
    count_emergency = df['emergency_route'].sum()
    print(f"    ✓ 緊急輸送道路: {count_emergency}件 ({count_emergency/len(df)*100:.1f}%)")
    
    return df

def extract_overpass_dummy(df):
    """跨線橋ダミー変数を抽出（○=1, それ以外=0）"""
    print("  🌉 跨線橋ダミー変数を抽出中...")
    
    if '跨線橋' not in df.columns:
        print("    ⚠ 「跨線橋」列が見つかりません。デフォルト値0を設定します。")
        df['overpass'] = 0
        return df
    
    df['overpass'] = df['跨線橋'].apply(lambda x: 1 if x == '○' else 0)
    
    count_overpass = df['overpass'].sum()
    print(f"    ✓ 跨線橋: {count_overpass}件 ({count_overpass/len(df)*100:.1f}%)")
    
    return df

def extract_geographic_features(df):
    """地理空間特徴量を追加: 桁下河川判定と海岸線までの距離"""
    print("  🗺️ 地理空間特徴量を抽出中...")
    
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import warnings
        warnings.filterwarnings('ignore')
        
        # 1. 橋梁座標データの確認
        if '緯度' not in df.columns or '経度' not in df.columns:
            print("    ⚠ 緯度・経度列が見つかりません。デフォルト値を設定します。")
            df['under_river'] = 0
            df['distance_to_coast_km'] = 10.0
            return df
        
        # 数値化して欠損値を確認
        df['緯度'] = pd.to_numeric(df['緯度'], errors='coerce')
        df['経度'] = pd.to_numeric(df['経度'], errors='coerce')
        
        # 有効な座標データのみ処理
        valid_coords = df[df['緯度'].notna() & df['経度'].notna()].copy()
        
        if len(valid_coords) == 0:
            print("    ⚠ 有効な座標データがありません。デフォルト値を設定します。")
            df['under_river'] = 0
            df['distance_to_coast_km'] = 10.0
            return df
        
        print(f"    ✓ 有効な座標: {len(valid_coords)}件 ({len(valid_coords)/len(df)*100:.1f}%)")
        
        # GeoDataFrameに変換
        gdf_bridges = gpd.GeoDataFrame(
            valid_coords,
            geometry=gpd.points_from_xy(valid_coords['経度'], valid_coords['緯度']),
            crs="EPSG:4326"  # WGS84
        )
        
        # 2. 河川データの読み込み
        river_file = config.RIVER_SHAPEFILE
        print(f"    📂 河川データ: {river_file}")
        
        try:
            rivers = gpd.read_file(river_file)
            
            # CRSが設定されていない場合はWGS84 (EPSG:4326)を設定
            # 国土数値情報のShapefileは座標がWGS84だがCRS未設定の場合がある
            if rivers.crs is None:
                rivers.set_crs("EPSG:4326", inplace=True)
            elif rivers.crs != "EPSG:4326":
                rivers = rivers.to_crs("EPSG:4326")
            
            print(f"    ✓ 河川データ読み込み: {len(rivers)}件")
            
            # 投影座標系に変換してバッファ作成（UTM Zone 53N: 山口県に適した座標系）
            rivers_proj = rivers.to_crs("EPSG:32653")  # UTM Zone 53N
            rivers_buffer = rivers_proj.buffer(50)  # 50m
            
            # バッファをWGS84に戻す
            rivers_buffer_wgs = rivers_buffer.to_crs("EPSG:4326")
            try:
                rivers_union = rivers_buffer_wgs.union_all()
            except AttributeError:
                # 古いgeopandasの場合はunary_unionを使用
                rivers_union = rivers_buffer_wgs.unary_union
            
            # 桁下河川判定
            gdf_bridges['under_river'] = gdf_bridges.geometry.apply(
                lambda point: 1 if rivers_union.contains(point) else 0
            )
            
            count_under_river = gdf_bridges['under_river'].sum()
            print(f"    ✓ 桁下河川あり: {count_under_river}件 ({count_under_river/len(gdf_bridges)*100:.1f}%)")
            
        except Exception as e:
            print(f"    ⚠ 河川データ読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            gdf_bridges['under_river'] = 0
        
        # 3. 海岸線データの読み込み
        coastline_file = config.COASTLINE_SHAPEFILE
        print(f"    📂 海岸線データ: {coastline_file}")
        
        try:
            coastline = gpd.read_file(coastline_file)
            
            # CRSが設定されていない場合はWGS84 (EPSG:4326)を設定
            if coastline.crs is None:
                coastline.set_crs("EPSG:4326", inplace=True)
            elif coastline.crs != "EPSG:4326":
                coastline = coastline.to_crs("EPSG:4326")
            
            print(f"    ✓ 海岸線データ読み込み: {len(coastline)}件")
            
            # 投影座標系で距離計算（より正確）
            gdf_bridges_proj = gdf_bridges.to_crs("EPSG:32653")  # UTM Zone 53N
            coastline_proj = coastline.to_crs("EPSG:32653")
            
            try:
                coastline_union = coastline_proj.union_all()
            except AttributeError:
                coastline_union = coastline_proj.unary_union
            
            # 海岸線までの距離計算（m単位をkmに変換）
            gdf_bridges['distance_to_coast_km'] = gdf_bridges_proj.geometry.apply(
                lambda point: coastline_union.distance(point) / 1000.0  # m→km
            )
            
            print(f"    ✓ 海岸線距離範囲: {gdf_bridges['distance_to_coast_km'].min():.2f}〜{gdf_bridges['distance_to_coast_km'].max():.2f}km")
            print(f"    ✓ 海岸線距離平均: {gdf_bridges['distance_to_coast_km'].mean():.2f}km")
            
        except Exception as e:
            print(f"    ⚠ 海岸線データ読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            gdf_bridges['distance_to_coast_km'] = 10.0
        
        # 4. 元のDataFrameにマージ
        df = df.merge(
            gdf_bridges[['under_river', 'distance_to_coast_km']],
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # 座標が無効だった行にデフォルト値を設定
        df['under_river'] = df['under_river'].fillna(0).astype(int)
        df['distance_to_coast_km'] = df['distance_to_coast_km'].fillna(
            df['distance_to_coast_km'].median() if df['distance_to_coast_km'].notna().any() else 10.0
        )
        
        print(f"    ✅ 地理空間特徴量の追加完了")
        
    except ImportError as e:
        print(f"    ⚠ geopandas/shapely未インストール: {e}")
        print("    → デフォルト値を設定します。")
        df['under_river'] = 0
        df['distance_to_coast_km'] = 10.0
    except Exception as e:
        print(f"    ⚠ 地理空間処理エラー: {e}")
        import traceback
        traceback.print_exc()
        df['under_river'] = 0
        df['distance_to_coast_km'] = 10.0
    
    return df

def normalize_repair_year(df):
    """最新補修年度をmin-max正規化（0〜1の範囲）"""
    print("  🔧 最新補修年度を正規化中...")
    
    if '最新補修年度' not in df.columns:
        print("    ⚠ 「最新補修年度」列が見つかりません。デフォルト値0.5を設定します。")
        df['repair_year_normalized'] = 0.5
        return df
    
    # 数値化
    df['最新補修年度'] = pd.to_numeric(df['最新補修年度'], errors='coerce')
    
    # 欠損値でない行のみでmin-max正規化
    valid_years = df['最新補修年度'].dropna()
    
    if len(valid_years) > 0:
        min_year = valid_years.min()
        max_year = valid_years.max()
        
        if max_year > min_year:
            df['repair_year_normalized'] = (df['最新補修年度'] - min_year) / (max_year - min_year)
        else:
            df['repair_year_normalized'] = 0.5
        
        # 欠損値は0（補修なし）として扱う
        df['repair_year_normalized'] = df['repair_year_normalized'].fillna(0.0)
        
        print(f"    ✓ 補修年度範囲: {int(min_year)}〜{int(max_year)}年")
        print(f"    ✓ 補修実施: {len(valid_years)}件 ({len(valid_years)/len(df)*100:.1f}%)")
    else:
        print("    ⚠ 有効な補修年度データがありません。デフォルト値0を設定します。")
        df['repair_year_normalized'] = 0.0
    
    return df

def extract_municipality_from_address(address):
    """所在地から市町村名を抽出"""
    if pd.isna(address):
        return None
    
    address_str = str(address)
    
    # 山口県の市町村リスト
    municipalities = [
        '下関市', '宇部市', '山口市', '萩市', '防府市', '下松市', '岩国市',
        '光市', '長門市', '柳井市', '美祢市', '周南市', '山陽小野田市',
        '周防大島町', '和木町', '上関町', '田布施町', '平生町', '阿武町'
    ]
    
    # 市町村名を検索
    for municipality in municipalities:
        if municipality in address_str:
            return municipality
    
    return None

def merge_municipal_data(bridge_df, fiscal_df, population_df):
    """市町村データを橋梁データに結合する"""
    print("🔗 市町村データを統合中...")
    
    # 市町村名の列を推定
    municipal_cols = ['市町村', '自治体', '市区町村', '管理者', '所在地']
    bridge_municipal_col = None
    
    for col in municipal_cols:
        if col in bridge_df.columns:
            bridge_municipal_col = col
            break
    
    if not bridge_municipal_col:
        print("  ⚠ 橋梁データに市町村名列が見つかりません。")
        # ダミーデータで代用
        bridge_df['population_decline'] = 10.0
        bridge_df['aging_rate'] = 30.0
        bridge_df['fiscal_index'] = 0.5
        return bridge_df
    
    # 「所在地」列の場合は市町村名を抽出
    if bridge_municipal_col == '所在地':
        print(f"  ✓ 「所在地」列から市町村名を抽出中...")
        bridge_df['市町村'] = bridge_df[bridge_municipal_col].apply(extract_municipality_from_address)
        
        # 抽出できなかった件数を表示
        null_count = bridge_df['市町村'].isna().sum()
        if null_count > 0:
            print(f"  ⚠ {null_count}件の橋梁で市町村名を抽出できませんでした")
        
        bridge_municipal_col = '市町村'
        
        # 市町村別の橋梁数を表示
        municipality_counts = bridge_df['市町村'].value_counts()
        print(f"  市町村別橋梁数（上位5件）:")
        for municipality, count in municipality_counts.head(5).items():
            print(f"    {municipality}: {count}件")
    
    # 財政力指数の処理
    fiscal_processed = process_fiscal_data(fiscal_df)
    
    # 人口統計の処理
    population_processed = process_population_data(population_df)
    
    # 市町村名で結合
    if fiscal_processed is not None:
        bridge_df = bridge_df.merge(fiscal_processed, 
                                     left_on=bridge_municipal_col, 
                                     right_on='municipality',
                                     how='left')
    else:
        bridge_df['fiscal_index'] = 0.5
        bridge_df['future_burden_ratio'] = 50.0
    
    if population_processed is not None:
        bridge_df = bridge_df.merge(population_processed,
                                     left_on=bridge_municipal_col,
                                     right_on='municipality',
                                     how='left')
        
        # 人口データがある市町村の5パーセンタイル値を計算（小規模町村想定）
        total_pop_5th = population_processed['total_population'].quantile(0.05)
        aging_rate_5th = population_processed['aging_rate'].quantile(0.05)
        
        print(f"  ✓ 人口5パーセンタイル値: 総人口={total_pop_5th:.0f}, 高齢化率={aging_rate_5th:.1f}%")
        
        # 欠損値を5パーセンタイル値で補完（小規模町村と想定）
        bridge_df['aging_rate'] = bridge_df['aging_rate'].fillna(aging_rate_5th)
    else:
        bridge_df['aging_rate'] = 30.0
    
    # 財政指標の欠損値を平均値で埋める
    bridge_df['fiscal_index'] = bridge_df['fiscal_index'].fillna(bridge_df['fiscal_index'].mean())
    bridge_df['future_burden_ratio'] = bridge_df['future_burden_ratio'].fillna(bridge_df['future_burden_ratio'].mean())
    
    return bridge_df

def process_fiscal_data(fiscal_df):
    """財政力指数と将来負担比率データを処理する"""
    try:
        # 山口県のデータのみ抽出
        if '都道府県名' in fiscal_df.columns:
            fiscal_df = fiscal_df[fiscal_df['都道府県名'] == '山口県'].copy()
            print(f"  ✓ 山口県のデータを抽出: {len(fiscal_df)}件")
        
        # 財政力指数と将来負担比率の列を探す
        fiscal_index_cols = ['財政力指数', '財政指数']
        future_burden_cols = ['将来負担比率', '将来負担率']
        municipal_cols = ['団体名', '市町村', '自治体名']
        
        fiscal_index_col = None
        future_burden_col = None
        municipal_col = None
        
        for col in fiscal_index_cols:
            if col in fiscal_df.columns:
                fiscal_index_col = col
                break
        
        for col in future_burden_cols:
            if col in fiscal_df.columns:
                future_burden_col = col
                break
        
        for col in municipal_cols:
            if col in fiscal_df.columns:
                municipal_col = col
                break
        
        if fiscal_index_col and future_burden_col and municipal_col:
            result = fiscal_df[[municipal_col, fiscal_index_col, future_burden_col]].copy()
            result.columns = ['municipality', 'fiscal_index', 'future_burden_ratio']
            result['fiscal_index'] = pd.to_numeric(result['fiscal_index'], errors='coerce')
            result['future_burden_ratio'] = pd.to_numeric(result['future_burden_ratio'], errors='coerce')
            
            # 欠損値を除外
            result = result.dropna(subset=['fiscal_index', 'future_burden_ratio'])
            
            print(f"  ✓ 財政指標を取得: {len(result)}市町村")
            print(f"    平均財政力指数: {result['fiscal_index'].mean():.3f}")
            print(f"    平均将来負担比率: {result['future_burden_ratio'].mean():.3f}%")
            
            return result
        else:
            print(f"  ⚠ 財政力指数の適切な列が見つかりません。")
            print(f"    利用可能な列: {fiscal_df.columns.tolist()[:10]}")
            return None
    except Exception as e:
        print(f"  ⚠ 財政データ処理エラー: {e}")
        return None

def process_population_data(population_df):
    """人口統計データを処理する"""
    try:
        # 山口県のデータのみ抽出
        if '都道府県名' in population_df.columns:
            population_df = population_df[population_df['都道府県名'] == '山口県'].copy()
        
        # 性別が「計」のデータのみ抽出（男女別を除外）
        if '性別' in population_df.columns:
            population_df = population_df[population_df['性別'] == '計'].copy()
        
        # 市町村名の列を探す
        municipal_cols = ['市区町村名', '市区町村', '市町村', '自治体名']
        municipal_col = None
        
        for col in municipal_cols:
            if col in population_df.columns:
                municipal_col = col
                break
        
        # 総数列の確認
        if '総数' not in population_df.columns:
            print(f"  ⚠ 人口統計の「総数」列が見つかりません。")
            print(f"    利用可能な列: {population_df.columns.tolist()[:15]}")
            return None
        
        if municipal_col:
            # 必要な列を選択
            cols_to_select = [municipal_col, '総数']
            
            # 65歳以上の年齢区分列
            elderly_cols = ['65歳～69歳', '70歳～74歳', '75歳～79歳', '80歳～84歳', 
                          '85歳～89歳', '90歳～94歳', '95歳～99歳', '100歳以上']
            
            # 存在する高齢者列を追加
            for col in elderly_cols:
                if col in population_df.columns:
                    cols_to_select.append(col)
            
            result = population_df[cols_to_select].copy()
            
            # 列名をリネーム
            rename_dict = {municipal_col: 'municipality', '総数': 'total_population'}
            result = result.rename(columns=rename_dict)
            
            result['total_population'] = pd.to_numeric(result['total_population'], errors='coerce')
            
            # 65歳以上人口を計算
            elderly_population = 0
            for col in elderly_cols:
                if col in result.columns:
                    elderly_population += pd.to_numeric(result[col], errors='coerce').fillna(0)
            
            # 高齢化率を計算
            result['aging_rate'] = (elderly_population / result['total_population'] * 100).fillna(30.0)
            
            # 山口県の市町村のみ抽出
            yamaguchi_municipalities = [
                '下関市', '宇部市', '山口市', '萩市', '防府市', '下松市', '岩国市',
                '光市', '長門市', '柳井市', '美祢市', '周南市', '山陽小野田市',
                '周防大島町', '和木町', '上関町', '田布施町', '平生町', '阿武町'
            ]
            
            # 市町村名を含むデータのみ抽出
            result = result[result['municipality'].isin(yamaguchi_municipalities)]
            
            print(f"  ✓ 人口統計を取得: {len(result)}市町村")
            print(f"    平均高齢化率: {result['aging_rate'].mean():.1f}%")
            
            return result
        else:
            print("  ⚠ 人口統計の市町村名列が見つかりません。")
            print(f"    利用可能な列: {population_df.columns.tolist()[:10]}")
            return None
    except Exception as e:
        print(f"  ⚠ 人口データ処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_all_data():
    """すべてのデータを前処理して統合する"""
    print("\n" + "="*60)
    print("📊 データ前処理を開始します")
    print("="*60 + "\n")
    
    # データ読み込み
    bridge_df = load_bridge_data()
    fiscal_df = load_fiscal_data()
    population_df = load_population_data()
    
    if bridge_df is None:
        print("\n❌ 橋梁データが読み込めませんでした。処理を中断します。")
        return None
    
    # 橋梁データの処理
    print("\n🔧 橋梁データを処理中...")
    bridge_df = calculate_bridge_age(bridge_df)
    bridge_df = extract_condition_score(bridge_df)
    bridge_df = calculate_maintenance_priority(bridge_df)
    
    # 新しい5つの特徴量を追加
    print("\n🆕 新規特徴量を追加中...")
    bridge_df = extract_structure_category(bridge_df)
    bridge_df = calculate_bridge_area(bridge_df)
    bridge_df = extract_emergency_route_dummy(bridge_df)
    bridge_df = extract_overpass_dummy(bridge_df)
    bridge_df = normalize_repair_year(bridge_df)
    
    # 地理空間特徴量を追加（v0.4: 桁下河川、海岸線距離）
    print("\n🗺️ 地理空間特徴量を追加中...")
    bridge_df = extract_geographic_features(bridge_df)
    
    # 市町村データと結合
    if fiscal_df is not None or population_df is not None:
        bridge_df = merge_municipal_data(bridge_df, fiscal_df, population_df)
    else:
        print("  ⚠ 市町村データがないため、デフォルト値を使用します。")
        bridge_df['population_decline'] = 10.0
        bridge_df['aging_rate'] = 30.0
        bridge_df['fiscal_index'] = 0.5
    
    # 特徴量カラムのみを抽出
    feature_cols = config.FEATURE_COLUMNS
    available_cols = [col for col in feature_cols if col in bridge_df.columns]
    
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"  ⚠ 以下の特徴量が不足しています: {missing}")
    
    # 結果を保存
    bridge_df.to_csv(config.PROCESSED_DATA_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ 前処理完了: {len(bridge_df)}件の橋梁データ")
    print(f"📁 保存先: {config.PROCESSED_DATA_FILE}")
    print(f"📋 特徴量: {', '.join(available_cols)}")
    
    return bridge_df

if __name__ == "__main__":
    preprocess_all_data()
