# -*- coding: utf-8 -*-
"""
クラスタリングメインスクリプト: 山口県橋梁維持管理クラスタリングMVP
- KMeansクラスタリング
- シルエットスコアによる最適クラスタ数決定
- PCAによる次元削減
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import config

def load_processed_data():
    """前処理済みデータを読み込む"""
    print("\n" + "="*60)
    print("📂 前処理済みデータを読み込み中...")
    print("="*60)
    
    try:
        df = pd.read_csv(config.PROCESSED_DATA_FILE)
        print(f"✓ データ読み込み完了: {len(df)}件")
        return df
    except FileNotFoundError:
        print("\n❌ 前処理済みデータが見つかりません。")
        print("先に data_preprocessing.py を実行してください。")
        return None
    except Exception as e:
        print(f"\n❌ データ読み込みエラー: {e}")
        return None

def prepare_features(df):
    """特徴量を準備する"""
    print("\n🔧 特徴量を準備中...")
    
    # 設定ファイルから特徴量カラムを取得
    feature_cols = config.FEATURE_COLUMNS
    
    # 利用可能な特徴量カラムを確認
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) == 0:
        print("❌ 特徴量カラムが見つかりません。")
        return None, None
    
    print(f"📋 使用する特徴量 ({len(available_cols)}個):")
    for col in available_cols:
        print(f"   - {col}")
    
    # 特徴量データを抽出
    X = df[available_cols].copy()
    
    # 欠損値を平均値で埋める
    X = X.fillna(X.mean())
    
    # 無限大値を除外
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    print(f"✓ 特徴量準備完了: {X.shape[0]}行 × {X.shape[1]}列")
    
    return X, available_cols

def standardize_features(X):
    """特徴量を標準化する"""
    print("\n📊 特徴量を標準化中...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("✓ 標準化完了")
    
    return X_scaled, scaler

def perform_pca(X_scaled):
    """主成分分析を実行する"""
    print("\n🔍 PCAによる次元削減中...")
    
    n_components = min(config.PCA_COMPONENTS, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"✓ PCA完了: {n_components}次元に削減")
    print(f"📈 説明された分散:")
    for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"   PC{i+1}: {ev:.2%} (累積: {cv:.2%})")
    
    return X_pca, pca

def find_optimal_clusters(X_scaled):
    """シルエットスコアで最適なクラスタ数を探索する"""
    print("\n🎯 最適なクラスタ数を探索中...")
    
    min_k = config.MIN_CLUSTERS
    max_k = min(config.MAX_CLUSTERS, len(X_scaled) - 1)
    
    silhouette_scores = []
    k_values = range(min_k, max_k + 1)
    
    best_k = min_k
    best_score = -1
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        
        print(f"   k={k}: シルエットスコア = {score:.4f}")
        
        if score > best_score:
            best_k = k
            best_score = score
    
    print(f"\n✅ 最適クラスタ数: k={best_k} (スコア: {best_score:.4f})")
    
    return best_k, best_score, silhouette_scores

def perform_clustering(X_scaled, n_clusters):
    """KMeansクラスタリングを実行する"""
    print(f"\n🎨 KMeansクラスタリングを実行中 (k={n_clusters})...")
    
    kmeans = KMeans(n_clusters=n_clusters, 
                   random_state=config.RANDOM_STATE,
                   n_init=10,
                   max_iter=300)
    
    labels = kmeans.fit_predict(X_scaled)
    
    # クラスタごとのサンプル数を確認
    unique, counts = np.unique(labels, return_counts=True)
    print("\n📊 クラスタ分布:")
    for cluster_id, count in zip(unique, counts):
        print(f"   クラスタ {cluster_id}: {count}件 ({count/len(labels)*100:.1f}%)")
    
    print("✓ クラスタリング完了")
    
    return kmeans, labels

def analyze_clusters(df, labels, feature_cols):
    """クラスタごとの特徴量を分析する"""
    print("\n📈 クラスタ特性を分析中...")
    
    df_with_cluster = df.copy()
    df_with_cluster['cluster'] = labels
    
    # クラスタごとの特徴量平均
    cluster_summary = df_with_cluster.groupby('cluster')[feature_cols].mean()
    
    print("\n📋 クラスタごとの特徴量平均:")
    print(cluster_summary.to_string())
    
    # クラスタ解釈
    print("\n🏷️ クラスタ解釈:")
    for cluster_id in cluster_summary.index:
        row = cluster_summary.loc[cluster_id]
        
        # 維持管理困難度を判定
        high_risk_factors = []
        
        if 'bridge_age' in feature_cols and row['bridge_age'] > 50:
            high_risk_factors.append("高橋齢")
        
        if 'condition_score' in feature_cols and row['condition_score'] >= 3:
            high_risk_factors.append("健全度低下")
        
        if 'maintenance_priority' in feature_cols and row['maintenance_priority'] > 100:
            high_risk_factors.append("高補修優先度")
        
        if 'population_decline' in feature_cols and row['population_decline'] > 15:
            high_risk_factors.append("人口減少")
        
        if 'aging_rate' in feature_cols and row['aging_rate'] > 35:
            high_risk_factors.append("高齢化")
        
        if 'fiscal_index' in feature_cols and row['fiscal_index'] < 0.5:
            high_risk_factors.append("財政力弱")
        
        risk_level = "🔴 高リスク" if len(high_risk_factors) >= 3 else \
                     "🟡 中リスク" if len(high_risk_factors) >= 2 else \
                     "🟢 低リスク"
        
        print(f"\n   クラスタ {cluster_id} {risk_level}")
        if high_risk_factors:
            print(f"     特徴: {', '.join(high_risk_factors)}")
    
    return df_with_cluster, cluster_summary

def save_results(df_with_cluster, cluster_summary):
    """結果を保存する"""
    print("\n💾 結果を保存中...")
    
    # クラスタ結果を保存
    df_with_cluster.to_csv(config.CLUSTER_RESULT_FILE, index=False, encoding='utf-8-sig')
    print(f"✓ クラスタ結果: {config.CLUSTER_RESULT_FILE}")
    
    # クラスタサマリーを保存
    cluster_summary.to_csv(config.CLUSTER_SUMMARY_FILE, encoding='utf-8-sig')
    print(f"✓ クラスタサマリー: {config.CLUSTER_SUMMARY_FILE}")

def main():
    """メイン処理"""
    print("\n" + "="*60)
    print("🚀 橋梁維持管理クラスタリング MVP")
    print("="*60)
    
    # 1. データ読み込み
    df = load_processed_data()
    if df is None:
        return
    
    # 2. 特徴量準備
    X, feature_cols = prepare_features(df)
    if X is None:
        return
    
    # 3. 標準化
    X_scaled, scaler = standardize_features(X)
    
    # 4. PCA
    X_pca, pca = perform_pca(X_scaled)
    
    # 5. 最適クラスタ数探索
    best_k, best_score, silhouette_scores = find_optimal_clusters(X_scaled)
    
    # 6. クラスタリング実行
    kmeans, labels = perform_clustering(X_scaled, best_k)
    
    # 7. クラスタ分析
    df_with_cluster, cluster_summary = analyze_clusters(df, labels, feature_cols)
    
    # 8. 結果保存
    save_results(df_with_cluster, cluster_summary)
    
    print("\n" + "="*60)
    print("✅ 処理完了！")
    print("="*60)
    print(f"\n📁 出力ファイル:")
    print(f"   - {config.CLUSTER_RESULT_FILE}")
    print(f"   - {config.CLUSTER_SUMMARY_FILE}")
    print(f"\n💡 次のステップ: visualization.py を実行して結果を可視化してください。")
    
    return df_with_cluster, cluster_summary, X_pca, labels

if __name__ == "__main__":
    main()
