# -*- coding: utf-8 -*-
"""
統合実行スクリプト: 山口県橋梁維持管理 Agentic Clustering v0.2
自己評価と改善を繰り返す賢いクラスタリングシステム
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_preprocessing
import visualization
from agentic_workflow import AgenticClusteringWorkflow
import config

def main():
    """全処理を順番に実行"""
    print("\n" + "="*70)
    print("🤖 橋梁維持管理 Agentic Clustering v0.2")
    print("="*70 + "\n")
    
    try:
        # ステップ1: データ前処理
        print("\n" + "─"*70)
        print("【ステップ 1/3】データ前処理")
        print("─"*70)
        df_processed = data_preprocessing.preprocess_all_data()
        
        if df_processed is None:
            print("\n❌ データ前処理に失敗しました。処理を中断します。")
            return False
        
        input("\n⏸️  続行するにはEnterキーを押してください...")
        
        # ステップ2: Agenticクラスタリング
        print("\n" + "─"*70)
        print("【ステップ 2/3】Agenticクラスタリング実行")
        print("─"*70)
        
        # 特徴量カラムを取得
        feature_cols = [col for col in config.FEATURE_COLUMNS if col in df_processed.columns]
        
        if len(feature_cols) == 0:
            print("\n❌ 特徴量カラムが見つかりません。処理を中断します。")
            return False
        
        # Agenticワークフローを実行
        workflow = AgenticClusteringWorkflow(df_processed, feature_cols)
        result = workflow.run(
            quality_threshold=config.QUALITY_THRESHOLD,
            overlap_threshold=config.OVERLAP_THRESHOLD
        )
        
        if result is None:
            print("\n❌ クラスタリングに失敗しました。処理を中断します。")
            return False
        
        # 結果を保存
        df_with_cluster = result['df_with_cluster']
        cluster_summary = result['cluster_summary']
        
        df_with_cluster.to_csv(config.CLUSTER_RESULT_FILE, index=False, encoding='utf-8-sig')
        cluster_summary.to_csv(config.CLUSTER_SUMMARY_FILE, encoding='utf-8-sig')
        
        print(f"\n💾 結果を保存しました:")
        print(f"   - {config.CLUSTER_RESULT_FILE}")
        print(f"   - {config.CLUSTER_SUMMARY_FILE}")
        
        # 改善ログを保存
        log_file = os.path.join(config.OUTPUT_DIR, 'agentic_improvement_log.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Agenticクラスタリング改善ログ\n")
            f.write("="*70 + "\n\n")
            f.write(f"最適クラスタリング手法: {result['clustering_method']}\n")
            f.write(f"最適次元削減手法: {result['dim_reduction_method']}\n\n")
            f.write("改善履歴:\n")
            for i, log in enumerate(result['improvement_log'], 1):
                f.write(f"{i}. {log}\n")
            f.write("\n評価スコア:\n")
            for key, value in result['evaluation_scores'].items():
                f.write(f"  {key}: {value}\n")
        
        print(f"   - {log_file}")
        
        input("\n⏸️  続行するにはEnterキーを押してください...")
        
        # ステップ3: 可視化
        print("\n" + "─"*70)
        print("【ステップ 3/3】結果の可視化")
        print("─"*70)
        
        # 3次元埋め込みデータを取得（UMAP使用時）
        embedding_3d = None
        if result['dim_reduction_method'] == 'UMAP' and 'embedding_3d' in result:
            embedding_3d = result['embedding_3d']
        
        visualization.main(
            dim_reduction_method=result['dim_reduction_method'],
            embedding=result['embedding'],
            embedding_3d=embedding_3d
        )
        
        # 完了メッセージ
        print("\n" + "="*70)
        print("✅ すべての処理が完了しました！")
        print("="*70)
        print("\n📁 結果は output/ フォルダに保存されています。")
        print("\n次のファイルを確認してください:")
        print("  🤖 agentic_improvement_log.txt - Agenticクラスタリング改善ログ")
        print("  📊 cluster_pca_scatter.png - 散布図（最適次元削減手法）")
        if embedding_3d is not None:
            print("  🌐 cluster_pca_scatter_3d.png - 3次元散布図（NEW!）")
        print("  🔥 cluster_heatmap.png - 特徴量ヒートマップ")
        print("  🌳 cluster_hierarchy.png - クラスタ階層構造図（NEW!）")
        print("  📡 cluster_radar.png - レーダーチャート")
        print("  📊 cluster_distribution.png - クラスタ分布")
        print("  📦 feature_boxplots.png - 箱ひげ図")
        print("  📝 cluster_report.txt - 分析レポート")
        print("\n🎯 Agenticクラスタリングにより、最適な手法が自動選択されました！")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  ユーザーにより処理が中断されました。")
        return False
    
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 分析が正常に完了しました！")
    else:
        print("\n💔 分析を完了できませんでした。")
    
    input("\n終了するにはEnterキーを押してください...")
