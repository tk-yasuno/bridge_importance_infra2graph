# OSM Grid Fetch システム - 利用ガイド

## 概要

山口市（~1,000 km²）などの大規模エリアのOSMデータを、レート制限を回避して取得するための4×4グリッド分割システムです。

## 使い方

### 基本的な使い方（2ステップ）

```bash
# ステップ1: グリッド分割でOSMデータを取得
python fetch_osm_grid.py

# ステップ2: マージされたネットワークでパイプラインを実行
python main.py --use-merged-network
```

### ステップ1の詳細: OSMデータ取得

`fetch_osm_grid.py`は以下の処理を自動実行します：

1. **市域ポリゴン取得** (Nominatim API)
   ```
   地名: "山口市, 山口県, 日本"
   ```

2. **4×4グリッド分割**
   - 外接矩形を16セルに分割
   - 市域と交差する部分のみ抽出

3. **セル単位でOSMデータ取得**
   - 各セルごとに以下を取得：
     - 道路ネットワーク（nodes + edges）
     - 建物（オプション）
     - バス停（オプション）
   - セル間で5秒待機（レート制限対策）
   - 失敗時は3回まで自動リトライ

4. **マージ処理**
   - 全セルのネットワークを統合
   - ノードIDを一意化
   - 重複ノード削除（1m精度）

**推定所要時間:** 15-30分（リトライ含む、道路ネットワークのみの場合）

**出力ファイル:**
```
output/bridge_importance/
├── osm_cells/
│   ├── cell_00_roads_nodes.gpkg    # セル0の道路ノード
│   ├── cell_00_roads_edges.gpkg    # セル0の道路エッジ
│   ├── cell_01_roads_nodes.gpkg
│   ├── cell_01_roads_edges.gpkg
│   └── ... (16セル分)
├── yamaguchi_merged_roads_nodes.gpkg  # マージされたノード（使用）
├── yamaguchi_merged_roads_edges.gpkg  # マージされたエッジ（使用）
└── grid_visualization.png            # グリッド可視化図
```

### ステップ2の詳細: メインパイプライン実行

`python main.py --use-merged-network`は、ステップ1で生成されたマージファイルを使用します。

**通常モードとの違い:**
- OSMへの新規リクエストなし（高速）
- マージされたネットワークを直接読み込み
- それ以外の処理は同一

## カスタマイズ

### グリッドサイズの変更

`fetch_osm_grid.py`を編集：

```python
# デフォルト: 4×4
grid_gdf = fetcher.make_grid_over_polygon(city_poly, n_rows=4, n_cols=4)

# より細かく分割（レート制限が厳しい場合）
grid_gdf = fetcher.make_grid_over_polygon(city_poly, n_rows=6, n_cols=6)  # 6×6 = 36セル
```

### 取得データの選択

`fetch_osm_grid.py`の`run_for_all_cells()`呼び出しを編集：

```python
stats = fetcher.run_for_all_cells(
    grid_gdf,
    fetch_roads=True,        # 道路ネットワーク（必須）
    fetch_buildings=True,    # 建物を取得（時間増加）
    fetch_bus_stops=True,    # バス停を取得（時間増加）
    inter_cell_delay=10      # セル間待機時間を10秒に変更
)
```

**注意:** 建物とバス停を有効にすると、所要時間が2-3倍に増加します。

### レート制限対策の調整

```python
# osm_grid_fetcher.py の OSMGridFetcher.__init__() 内
ox.settings.timeout = 180  # タイムアウトを3分に設定（デフォルト）
ox.settings.use_cache = True  # キャッシュ有効化（推奨）

# fetch_roads_for_cell() などのメソッド呼び出し時
retry_count=3      # リトライ回数（デフォルト）
retry_delay=10     # リトライ間隔（秒）
```

## トラブルシューティング

### 1. 一部のセルが失敗する

**症状:** 
```
[Cell 05] Roads fetch failed after 3 attempts
```

**対処法1**: 失敗したセルのみ再取得

`fetch_osm_grid.py`を編集して特定セルのみ処理：

```python
# 全セル処理の代わりに
failed_cells = grid_gdf[grid_gdf['cell_id'].isin([5, 12])]  # 失敗したセルのみ
stats = fetcher.run_for_all_cells(failed_cells, ...)
```

**対処法2**: 待機時間を増やす

```python
stats = fetcher.run_for_all_cells(
    grid_gdf,
    inter_cell_delay=15  # 5秒 → 15秒に増加
)
```

### 2. マージされたファイルが見つからない

**症状:**
```
FileNotFoundError: Merged road network not found
```

**対処法:**

1. ステップ1を完了しているか確認：
   ```bash
   ls output/bridge_importance/yamaguchi_merged_*.gpkg
   ```

2. 出力ディレクトリのパスが正しいか確認（`config.yaml`）

3. ファイルが破損していないか確認：
   ```python
   import geopandas as gpd
   nodes = gpd.read_file('output/bridge_importance/yamaguchi_merged_roads_nodes.gpkg')
   print(f"Nodes: {len(nodes)}")
   ```

### 3. メモリ不足エラー

**症状:**
```
MemoryError: Unable to allocate array
```

**対処法:**

1. グリッドをより細かく分割（4×4 → 6×6）
2. Pythonの実行を64bit版で行う
3. 不要な列を削除してファイルサイズを削減

### 4. グリッドの可視化が失敗する

**症状:**
```
Grid visualization failed: ...
```

**影響:** 可視化のみが失敗、データ取得は正常に完了
**対処法:** `matplotlib`のインストール確認、またはグラフ生成をスキップ

## 出力の検証

### マージされたネットワークの統計確認

```python
import geopandas as gpd

# ノードとエッジを読み込み
nodes = gpd.read_file('output/bridge_importance/yamaguchi_merged_roads_nodes.gpkg')
edges = gpd.read_file('output/bridge_importance/yamaguchi_merged_roads_edges.gpkg')

print(f"Total nodes: {len(nodes)}")
print(f"Total edges: {len(edges)}")
print(f"Cells processed: {nodes['cell_id'].nunique()}")

# 重複チェック
print(f"Duplicate nodes: {nodes.duplicated().sum()}")
print(f"Nodes per cell: {nodes.groupby('cell_id').size()}")
```

**期待される出力例:**
```
Total nodes: 18231
Total edges: 24132
Cells processed: 16
Duplicate nodes: 0
Nodes per cell: 
cell_id
0     1523
1     1387
2     1094
...
```

### グリッド分割の可視化確認

生成された`grid_visualization.png`を確認：
- 赤線: 山口市境界
- 青線: グリッドセル境界
- 黄色ラベル: セルID（0-15）

## パフォーマンス

### 標準的な実行時間（山口市 ~1,000 km²）

| 処理 | 時間 | 備考 |
|------|------|------|
| 市域ポリゴン取得 | 5秒 | Nominatim API |
| グリッド生成 | 1秒 | 計算処理 |
| **道路データ取得** | **12-20分** | 16セル × 5秒待機 + リトライ |
| 建物データ取得 | 10-15分 | オプション |
| バス停データ取得 | 5-10分 | オプション |
| **マージ処理** | **30秒** | ノードID一意化、重複削除 |
| **合計（道路のみ）** | **15-25分** | 推奨設定 |
| 合計（全データ） | 30-50分 | 建物+バス停含む |

### ネットワーク使用量

- 道路ネットワーク: 約50-100 MB（16セル）
- 建物: 約200-500 MB（密集地域で増加）
- バス停: 約1-5 MB

## ベストプラクティス

### 推奨設定（山口市規模）

```python
# fetch_osm_grid.py
grid_gdf = fetcher.make_grid_over_polygon(city_poly, n_rows=4, n_cols=4)

stats = fetcher.run_for_all_cells(
    grid_gdf,
    fetch_roads=True,        # 必須
    fetch_buildings=False,   # スキップ（高速化）
    fetch_bus_stops=False,   # スキップ（高速化）
    inter_cell_delay=5       # 標準
)
```

**理由:**
- 橋梁重要度評価には道路ネットワークが最重要
- 建物・バス停は別の手段で取得可能（より高速なAPI）
- 4×4グリッドで十分な粒度

### 夜間実行の推奨

大規模データ取得は夜間実行がおすすめ：

```bash
# Linux/Mac
nohup python fetch_osm_grid.py > osm_fetch.log 2>&1 &

# Windows PowerShell
Start-Process python -ArgumentList "fetch_osm_grid.py" -NoNewWindow -RedirectStandardOutput "osm_fetch.log"
```

## 関連ファイル

- `osm_grid_fetcher.py` - グリッド分割取得の実装
- `fetch_osm_grid.py` - 実行スクリプト
- `graph_builder.py` - マージネットワーク読み込み機能
- `main.py` - `--use-merged-network`オプション対応

## 参考情報

- OSMnx Documentation: https://osmnx.readthedocs.io/
- Overpass API rate limits: https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances
- GeoPackage format: https://www.geopackage.org/
