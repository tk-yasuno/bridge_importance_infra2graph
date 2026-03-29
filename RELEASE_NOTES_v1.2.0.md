# Release Notes - v1.2.0: Bridge Closure Impact Simulation

**Released:** 2026-03-29  
**Version:** 1.2.0  
**Status:** Stable MVP

## 概要

v1.2.0では、**橋梁閉鎖シナリオシミュレーション**機能を追加しました。「この橋が使えなくなったらどうなる？」という重要な問いに、定量的な答えを提供します。

## 新機能

### 1. Bridge Closure Impact Simulator

橋梁を1つずつ削除し、交通ネットワークへの影響を定量評価するシミュレーションシステムです。

#### 評価指標

**主要指標（Primary Metrics）:**
- **平均最短経路長の変化**: 橋梁閉鎖後の迂回距離の増加
- **到達不能ノード数**: 閉鎖により孤立するネットワークノードの数
- **到達不能バス停数**: アクセスできなくなる公共交通拠点の数

**派生指標（Derived Metrics）:**
- 経路長増加率（%）
- ノード損失率（%）
- バス停損失率（%）
- 連結成分数の変化

#### 技術仕様

**アルゴリズム:**
```
For each target bridge:
  1. Copy original graph G
  2. Remove bridge node and its edges
  3. Compute connected components
  4. Calculate shortest paths (sampled)
  5. Count accessible bus stops
  6. Compute impact deltas vs baseline
```

**パフォーマンス:**
- サンプリングベース最短経路計算（デフォルト: 500ノード）
- 処理速度: 約2秒/橋（132橋で約4-5分）
- メモリ使用量: 約200-500 MB

**制約:**
- 対象: Low以上の重要度（132橋）、Very Lowは除外（計算量削減）
- グラフサイズ: 18,000ノード、24,000エッジ規模で検証済み

### 2. 新規モジュール

#### `bridge_closure_simulator.py` (600行)

**主要クラス:**

```python
class BridgeClosureSimulator:
    """橋梁閉鎖シミュレータ"""
    
    def compute_baseline_metrics(self, sample_size: int) -> Dict
        """ベースラインメトリクス計算"""
    
    def simulate_bridge_closure(self, bridge_id: str) -> Dict
        """単一橋梁の閉鎖シミュレーション"""
    
    def simulate_multiple_bridges(self, bridge_ids: List[str]) -> pd.DataFrame
        """複数橋梁の一括シミュレーション"""
    
    def generate_impact_report(self, results_df: pd.DataFrame, output_path: Path)
        """影響度レポート生成（Markdown）"""
```

**機能詳細:**
- ノードタイプ自動分類（bridge, street, bus_stop）
- サンプリングベース最短経路計算（計算量削減）
- 連結成分分析（孤立ノード検出）
- プログレスバー付き一括処理

#### `run_closure_simulation.py` (340行)

**実行スクリプト:**

```bash
# 基本実行（Low以上132橋、サンプルサイズ500）
python run_closure_simulation.py

# サンプルサイズ増加（精度向上、時間増加）
python run_closure_simulation.py --sample-size 1000

# Very Low含む全橋梁対象
python run_closure_simulation.py --include-very-low
```

**処理フロー:**
1. データ読み込み（bridge_importance_scores.geojson, heterogeneous_graph.pkl）
2. 重要度フィルタリング（Very Low除外）
3. シミュレーター初期化＆ベースライン計算
4. 閉鎖シミュレーション実行（進捗表示）
5. 結果保存（CSV, Markdown, PNG）

### 3. 出力ファイル

**`output/bridge_importance/closure_simulation/`:**

| ファイル | 形式 | 内容 |
|---------|------|-----|
| `closure_simulation_results.csv` | CSV | 全橋梁のシミュレーション結果（数値データ） |
| `closure_impact_report.md` | Markdown | 詳細レポート（ランキング、統計、全リスト） |
| `closure_impact_distribution.png` | PNG | 影響度分布の可視化（4サブプロット） |
| `closure_impact_top10.png` | PNG | トップ10橋梁の棒グラフ（3指標） |

**CSV列:**
- `bridge_id`: 橋梁ID
- `bridge_node`: グラフ内のノードID
- `avg_shortest_path_after`: 閉鎖後の平均最短経路長
- `num_connected_nodes_after`: 閉鎖後の連結ノード数
- `accessible_bus_stops_after`: 閉鎖後のアクセス可能バス停数
- `delta_avg_shortest_path`: 経路長の変化量
- `delta_connected_nodes`: 連結ノードの変化量（負=損失）
- `delta_accessible_bus_stops`: バス停の変化量（負=損失）
- `pct_path_increase`: 経路長増加率（%）
- `pct_nodes_lost`: ノード損失率（%）
- `pct_bus_stops_lost`: バス停損失率（%）

### 4. 可視化

#### Distribution Plot (`closure_impact_distribution.png`)

4つのヒストグラム:
1. **経路長増加の分布**: 閉鎖による迂回距離の増加
2. **ノード損失の分布**: 孤立するノードの数
3. **バス停損失の分布**: アクセス不能になるバス停
4. **経路長増加率の分布**: %での影響度

#### Top 10 Plot (`closure_impact_top10.png`)

3つの水平棒グラフ:
1. **Path Length Increase Top 10**: 最も迂回距離が増える橋梁
2. **Node Loss Top 10**: 最も多くのノードを孤立させる橋梁
3. **Bus Stop Loss Top 10**: 最も多くのバス停をアクセス不能にする橋梁

### 5. Markdown レポート

**構成:**
- **Baseline Metrics**: 閉鎖前の基準値
- **Simulation Summary**: 全体統計（平均、標準偏差）
- **Top 10 Rankings**: 3指標別のランキングテーブル
  - By Average Shortest Path Increase
  - By Nodes Lost
  - By Bus Stops Lost
- **Impact Distribution**: 影響度の分類（High/Medium/Low）
- **Full Simulation Results**: 全橋梁の結果リスト

## アーキテクチャ

### データフロー

```
[main.py実行済み]
    ↓
  bridge_importance_scores.geojson (重要度スコア付き橋梁データ)
  heterogeneous_graph.pkl (異種グラフ)
    ↓
[run_closure_simulation.py]
    ↓
  1. フィルタリング: Low以上の132橋を抽出
  2. シミュレーター初期化
  3. ベースライン計算（閉鎖前）
  4. 各橋梁を削除してシミュレーション
  5. 影響度計算（差分）
    ↓
  closure_simulation_results.csv
  closure_impact_report.md
  closure_impact_distribution.png
  closure_impact_top10.png
```

### 計算効率化

**サンプリング戦略:**
- 全ノードペアの最短経路計算は O(n³) で非現実的
- ランダムサンプリング（デフォルト500ノード）で O(n·s·log n) に削減
- 精度: サンプルサイズ500で±5%の誤差範囲

**並列化オプション（将来拡張）:**
```python
# 現在: シーケンシャル実行
for bridge_id in bridge_ids:
    result = simulate_bridge_closure(bridge_id)

# 将来: マルチプロセス対応
from multiprocessing import Pool
with Pool(4) as pool:
    results = pool.map(simulate_bridge_closure, bridge_ids)
```

## 使用例

### 基本的な使い方

```bash
# ステップ1: v1.0パイプラインで重要度スコア生成
python main.py --use-merged-network

# ステップ2: v1.2閉鎖シミュレーション実行
python run_closure_simulation.py
```

### カスタマイズ例

```bash
# 精度を上げる（計算時間2倍）
python run_closure_simulation.py --sample-size 1000

# 全橋梁を対象（Very Low含む、計算時間10倍）
python run_closure_simulation.py --include-very-low --sample-size 300
```

### プログラムからの利用

```python
import pickle
from bridge_closure_simulator import BridgeClosureSimulator

# グラフ読み込み
with open('output/bridge_importance/heterogeneous_graph.pkl', 'rb') as f:
    G = pickle.load(f)

# シミュレーター初期化
simulator = BridgeClosureSimulator(G)

# ベースライン計算
baseline = simulator.compute_baseline_metrics(sample_size=500)

# 単一橋梁の閉鎖シミュレーション
result = simulator.simulate_bridge_closure('BR_0530')
print(f"Path increase: {result['delta_avg_shortest_path']:.2f}")
print(f"Nodes lost: {abs(result['delta_connected_nodes'])}")

# 複数橋梁の一括シミュレーション
bridge_ids = ['BR_0530', 'BR_0533', 'BR_0001']
results_df = simulator.simulate_multiple_bridges(bridge_ids, sample_size=500)
print(results_df[['bridge_id', 'delta_avg_shortest_path', 'pct_path_increase']])
```

## パフォーマンス

### 計算時間（山口市791橋データセット）

| 対象橋梁数 | サンプルサイズ | 推定時間 |
|-----------|---------------|---------|
| 132（Low以上） | 500 | 4-5分 |
| 132（Low以上） | 1000 | 8-10分 |
| 791（全橋梁） | 500 | 25-30分 |

**環境:** Intel Core i7-12700, 32GB RAM, Python 3.12

### メモリ使用量

- ベースライン: 200 MB（グラフ読み込み）
- ピーク: 500 MB（シミュレーション中）

### スケーラビリティ

| グラフ規模 | ノード数 | エッジ数 | 処理可能性 |
|----------|---------|---------|----------|
| 小規模 | <5,000 | <10,000 | 高速（<1分） |
| 中規模（山口市） | 18,000 | 24,000 | 実用的（5分） |
| 大規模 | >50,000 | >100,000 | 要最適化 |

## 制限事項

1. **計算量の制約**
   - 全ノードペアの厳密な最短経路計算は非現実的
   - サンプリングベースで近似（デフォルト500ノード）

2. **対象橋梁の制限**
   - デフォルトでVery Lowカテゴリ（659橋）を除外
   - 計算時間とのトレードオフ

3. **バス停データの依存性**
   - OSMから取得したバス停データのみ対象
   - 実際の路線情報は含まない

4. **静的シミュレーション**
   - 動的な交通量は考慮しない
   - グラフトポロジーのみで評価

## トラブルシューティング

### エラー: `FileNotFoundError: bridge_importance_scores.geojson`

**原因:** v1.0パイプライン未実行

**解決策:**
```bash
python main.py --use-merged-network
```

### エラー: `FileNotFoundError: heterogeneous_graph.pkl`

**原因:** グラフが保存されていない

**解決策:** 同上（main.py実行）

### 警告: `Original graph is not fully connected`

**影響:** 一部のノードが孤立している（正常動作）

**対処:** 自動的に最大連結成分を使用（ログに明示）

### パフォーマンスが遅い

**対処法:**
1. サンプルサイズを減らす: `--sample-size 300`
2. Very Lowを除外: デフォルト動作
3. 並列化（将来実装）

## 今後の拡張計画

### v1.2.1（計画中）
- [ ] マルチプロセス並列化（4倍高速化）
- [ ] 進捗保存・レジューム機能
- [ ] GeoJSON出力（地図化対応）

### v1.3（検討中）
- [ ] 複数橋梁同時閉鎖シナリオ
- [ ] 交通量重み付き評価
- [ ] 時系列シミュレーション（復旧時間）
- [ ] 迂回ルートの可視化

### v2.0（長期ビジョン）
- [ ] 機械学習による影響度予測
- [ ] リアルタイム交通データ統合
- [ ] GUIベースのシナリオエディタ

## 関連ファイル

**新規追加:**
- `bridge_closure_simulator.py` - シミュレータ実装
- `run_closure_simulation.py` - 実行スクリプト
- `RELEASE_NOTES_v1.2.0.md` - このファイル

**更新:**
- `VERSION` - 1.1.0 → 1.2.0
- `README_JP.md` - v1.2セクション追加（次回更新予定）
- `QUICKSTART.md` - v1.2実行手順追加（次回更新予定）

## チェンジログ

**Added:**
- Bridge closure impact simulation system
- Baseline metrics computation
- Multiple impact indicators (path length, node loss, bus stop loss)
- Markdown report generation with rankings
- Distribution and Top 10 visualizations
- Command-line interface with options

**Changed:**
- VERSION 1.1.0 → 1.2.0

**Technical:**
- NetworkX-based graph manipulation
- Sampling-based shortest path computation
- Connected component analysis
- Progress bar integration (tqdm)

## 引用・クレジット

**Dependencies:**
- NetworkX 3.6.1 - Graph analysis
- Pandas / GeoPandas - Data handling
- Matplotlib / Seaborn - Visualization
- tqdm - Progress bars

**Contributors:**
- Yamaguchi City Bridge Dataset (791 bridges)
- OSM Contributors (Road network, bus stops)

## ライセンス

MIT License - 詳細は LICENSE ファイルを参照

## サポート

**Issues:** GitHub Issues  
**Documentation:** README_JP.md, QUICKSTART.md, このファイル  
**Contact:** プロジェクト管理者

---

**Next Release:** v1.2.1 (Planned: 2026-04-15)  
**Focus:** Parallel processing, Resume capability
