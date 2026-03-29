# Bridge Importance Scoring MVP - クイックスタートガイド

## 最速セットアップ（5分）

### 1. 環境構築

```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. データ確認

以下のデータが配置されているか確認：

- `data/Bridge_xy_location/YamaguchiPrefBridgeListOpen251122_154891.xlsx`
- `data/RiverDataKokudo/W05-08_35_GML/` （シェープファイル）
- `data/KaigansenDataKokudo/C23-06_35_GML/` （シェープファイル）

### 3. 実行

```bash
# 方法1: セットアップスクリプト経由（小規模エリア）
python setup_and_run.py

# 方法2: 直接実行（小規模エリア）
python main.py
```

### 3-A. 大規模エリア（山口市全域など）の場合

山口市（~1,000 km²）のような大規模エリアでは、OSM APIのレート制限により通常モードが失敗する可能性があります。
この場合、**グリッド分割モード**を使用してください：

```bash
# ステップ1: 4×4グリッド分割でOSMデータを取得（15-30分）
python fetch_osm_grid.py

# ステップ2: マージされたネットワークで実行（5-10分）
python main.py --use-merged-network
```

**グリッドモードの利点:**
- レート制限を回避（セル間5秒待機）
- 各セルは自動的に3回までリトライ
- 失敗したセルのみ再実行可能
- 密な結合グラフを自動生成

## 実行時間の目安

- データ読み込み: 10秒
- 異種グラフ構築（OSM取得含む）: 3-5分
- 媒介中心性計算: 2-10分（橋梁数・グラフ規模による）
- スコアリングと出力: 30秒

**合計: 約5-15分**

## トラブルシューティング

### OSMデータ取得がタイムアウトする場合

`config.yaml`の解析範囲を縮小：

```yaml
data:
  osm:
    bbox:
      min_lon: 131.4  # より狭い範囲に
      min_lat: 34.15
      max_lon: 131.5
      max_lat: 34.25
```

### メモリ不足エラー

`config.yaml`で近似計算を有効化：

```yaml
centrality:
  k: 100  # サンプル数を制限
```

### 座標系エラー

橋梁Excelファイルの座標カラム名を確認し、`data_loader.py`の`_find_column()`メソッドを調整。

## 結果の確認

出力ファイル（`output/bridge_importance/`）：

1. **bridge_importance_scores.csv** - 全橋梁のスコア（Excelで開ける）
2. **interactive_bridge_map.html** - 対話的地図（ブラウザで開く）
3. **bridge_importance_report.md** - 詳細レポート（テキストエディタで開く）
4. **top10_critical_bridges.csv** - トップ10橋梁の詳細

## v1.2 新機能: 橋梁閉鎖影響シミュレーション 🆕

**前提**: `main.py` を実行済みであること（異種グラフの生成が必要）

### クイック実行

```bash
# デフォルト設定で実行（Low以上の132橋梁、4-5分）
python run_closure_simulation.py
```

### カスタム設定

```bash
# サンプリングサイズの変更（精度 vs 速度のトレードオフ）
python run_closure_simulation.py --sample-size 1000  # より高精度（遅い）
python run_closure_simulation.py --sample-size 200   # より高速（低精度）

# 全橋梁を対象にする（Very Lowも含む791橋、25-30分）
python run_closure_simulation.py --include-very-low
```

### 出力ファイル（v1.2）

シミュレーション結果は `output/bridge_importance/closure_simulation/` に生成されます：

1. **closure_simulation_results.csv**
   - 全橋梁の閉鎖影響データ（11列）
   - Excelで開いてソート・フィルタ可能
   - 影響の大きい橋梁を特定

2. **closure_impact_report.md**
   - Markdownレポート
   - Top 10 ランキング（3つの影響指標別）
   - 統計サマリー

3. **closure_impact_distribution.png**
   - 4つの分布プロット
   - 影響指標のばらつきを可視化

4. **closure_impact_top10.png**
   - Top 10 ランキングのバーチャート
   - 経路長・ノード孤立・バス停影響の3軸

### 結果の解釈

**影響指標の意味:**

1. **delta_avg_shortest_path_pct** (平均経路長変化率)
   - 例: 5.2% → この橋を閉鎖すると、市民の平均移動距離が5.2%増加
   - 用途: 交通混雑・燃料費増加の推定

2. **delta_connected_nodes_pct** (接続ノード減少率)
   - 例: 2.8% → 道路網の2.8%が到達不能になる
   - 用途: 孤立地域の特定

3. **delta_accessible_bus_stops_pct** (バス停減少率)
   - 例: 4.1% → バス停の4.1%が利用不可に
   - 用途: 公共交通への影響評価

**活用例:**
- 災害対応計画: 「橋梁Xが使えない場合、移動距離8%増」
- 維持管理優先度: 閉鎖影響の大きい橋梁を優先修繕
- 予算折衝: 定量的根拠での説明資料
- 迂回路計画: 代替経路の事前検討

### 処理時間の目安

| 対象橋梁数 | サンプルサイズ | 処理時間 |
|-----------|--------------|---------|
| 132 (Low+) | 500 (default) | 4-5分 |
| 132 (Low+) | 1000 (高精度) | 8-10分 |
| 791 (全橋) | 500 (default) | 25-30分 |

**注**: 処理時間はPC性能に依存します。進捗バーで残り時間を確認できます。

## 次のステップ

### 可視化の追加生成

```python
from visualization import visualize_results
from utils import load_saved_results
import yaml

# 保存済みデータの読み込み
bridges, graph, metadata = load_saved_results('output/bridge_importance')

# 設定ファイルの読み込み
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 可視化の生成
visualize_results(bridges, config)
```

### 中心性指標の比較

```python
from utils import compare_centrality_measures

bridge_nodes = bridges['bridge_id'].tolist()
centrality_comparison = compare_centrality_measures(graph, bridge_nodes, limit=20)
```

### GISソフトへのエクスポート

```python
from utils import export_for_gis

# Shapefileに変換
export_for_gis(bridges, 'output/bridges.shp', format='shapefile')

# QGISなどで開ける形式
export_for_gis(bridges, 'output/bridges.gpkg', format='gpkg')
```

## 設定のカスタマイズ

### スコアリングの重みを変更

`config.yaml`：

```yaml
scoring:
  weights:
    betweenness: 0.7      # 媒介中心性の重み（デフォルト0.6）
    public_access: 0.2    # 公共施設アクセス
    traffic_volume: 0.1   # 交通量代理
```

### 近接関係の閾値を変更

```yaml
graph:
  proximity:
    bridge_to_building: 1500  # デフォルト1000m → 1500mに拡大
    bridge_to_bus_stop: 1000  # デフォルト800m → 1000mに拡大
```

## よくある質問

**Q: City2Graphがインストールできない**

A: 本MVPはOSMnxをベースに設計しており、City2Graphは補助的な役割です。OSMnxのみで実行可能です。

**Q: 実行が遅い**

A: `config.yaml`で`centrality.k`を設定し、近似計算を使用してください。また、OSM取得範囲を縮小することも有効です。

**Q: 橋梁の座標が正しく読み込まれない**

A: Excelファイルのカラム名を確認し、`data_loader.py`の`_find_column()`で適切なカラム名を追加してください。

## サポート

詳細なドキュメント: [README.md](README.md)

---

**MVP Version 1.0** | 2024年3月28日
