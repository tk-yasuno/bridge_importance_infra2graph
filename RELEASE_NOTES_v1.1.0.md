# Bridge Importance Scoring MVP - Release Notes v1.1.0

**リリース日**: 2026年3月29日  
**バージョン**: 1.1.0  
**リリースタイプ**: 機能追加（HGNN統合）

---

## 🎉 新機能：異種グラフニューラルネットワーク（HGNN）統合

v1.1 では、**PyTorch Geometric**を統合し、**深層学習による橋梁重要度予測**機能を追加しました。従来の NetworkX 媒介中心性に加えて、グラフニューラルネットワーク（GNN）による予測モデルを構築できます。

### 主な追加機能

#### 1. HeteroData 変換モジュール (`hetero_data_converter.py`)

- NetworkX 異種グラフを PyTorch Geometric の `HeteroData` 形式に変換
- ノードタイプごとの特徴量抽出
  - **Bridge ノード (25特徴)**:
    - 健全度区分（Ⅰ-Ⅳ、one-hot encoded）
    - 橋齢、橋長、幅員
    - 河川・海岸線からの距離（対数スケール）
    - 既存の媒介中心性スコア
    - 周辺施設数（建物、病院、学校、公共施設）
    - Binary flags: 離島架橋、長大橋、特殊橋、重要物流道路、緊急輸送道路、跨線橋、跨道橋
  - **Street ノード**: 正規化座標 (x, y)
  - **Building ノード**: 建物カテゴリ（one-hot encoded）
  - **Bus Stop ノード**: プレースホルダー（将来拡張用）

- エッジタイプの自動抽出
  - `(bridge, to, street)`, `(bridge, to, building)`, `(street, to, street)` など

#### 2. HGNN モデル定義 (`hgnn_model.py`)

- **BridgeImportanceHGNN**: 標準モデル（2層 HeteroConv）
  - HeteroConv + GATConv（Graph Attention Networks）
  - HeteroConv + GraphSAGE
  - マルチヘッドアテンション（デフォルト: 4 heads）
  - ドロップアウト（0.2）
  - ノードタイプごとのエンコーディング層

- **BridgeImportanceHGNN_Simple**: 簡易モデル（1層 HeteroConv）
  - 小規模データや高速プロトタイピング用

- **タスク**: ノード回帰（Bridge ノードの重要度スコア予測）

#### 3. トレーニングパイプライン (`train_hgnn.py`)

- **データ分割**:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

- **トレーニング機能**:
  - MSE 損失関数
  - Adam オプティマイザ
  - Early stopping（patience=20）
  - 学習率: 0.001、Weight decay: 5e-4

- **評価指標**:
  - MSE（Mean Squared Error）
  - MAE（Mean Absolute Error）
  - RMSE（Root Mean Squared Error）
  - R²（決定係数）

- **可視化**:
  - トレーニング履歴（Loss/MAE曲線）
  - 予測値 vs 真値の散布図
  - 結果の CSV エクスポート

#### 4. データ変換スクリプト (`convert_to_heterodata.py`)

- 既存の NetworkX グラフ（`heterogeneous_graph.pkl`）と GeoDataFrame（`bridge_importance_scores.geojson`）から HeteroData を生成
- ノードタイプの自動検出と分類
- HeteroData の保存（`.pt` 形式）

---

## 📋 使い方

### ステップ 0: 既存パイプラインの実行（v1.0 機能）

```bash
# まず、v1.0 のパイプラインを実行して NetworkX グラフと重要度スコアを生成
python main.py
```

**出力**:
- `output/bridge_importance/heterogeneous_graph.pkl` - NetworkX グラフ
- `output/bridge_importance/bridge_importance_scores.geojson` - 重要度スコア付き橋梁データ

### ステップ 1: HeteroData への変換

```bash
python convert_to_heterodata.py
```

**出力**:
- `output/bridge_importance/heterogeneous_graph_heterodata.pt` - PyTorch Geometric HeteroData

**実行時間**: 約10-30秒

### ステップ 2: HGNN モデルのトレーニング

```bash
python train_hgnn.py
```

**出力**:
- `output/hgnn_training/best_hgnn_model.pt` - 学習済みモデル
- `output/hgnn_training/training_history.png` - Loss/MAE 曲線
- `output/hgnn_training/predictions_vs_truth.png` - 予測散布図
- `output/hgnn_training/test_metrics.csv` - テストメトリクス
- `output/hgnn_training/training_history.csv` - 履歴CSV

**実行時間**: 約5-15分（GPU使用時は1-3分）

---

## ⚙️ 設定

`config.yaml` に新しい `hgnn:` セクションを追加しました：

```yaml
hgnn:
  # モデルタイプ
  model_type: "standard"  # "standard" or "simple"
  
  # ハイパーパラメータ
  hidden_channels: 64
  num_layers: 2
  conv_type: "GAT"  # "GAT" or "SAGE"
  dropout: 0.2
  heads: 4  # GAT attention heads
  
  # トレーニング設定
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0005
  patience: 20
  
  # データ分割
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
```

---

## 📦 依存関係の追加

`requirements.txt` を更新し、以下のパッケージを追加しました：

```
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.17
scikit-learn>=1.0.0
```

**インストール**:

```bash
pip install -r requirements.txt
```

**注意**: PyTorch Geometric のインストールは環境依存です。詳細は [公式ドキュメント](https://pytorch-geometric.readthedocs.io/) を参照してください。

---

## 🔬 技術詳細

### モデルアーキテクチャ

#### HGNN モデル全体像

```mermaid
flowchart TD
    Bridge["Bridge\\n791, 25dim"]
    Street["Street\\nN, 2dim"]
    Building["Building\\nM, 5dim"]
    BusStop["Bus Stop\\nK, 2dim"]
    
    EncB["Bridge Encoder\\nLinear 25 to 64"]
    EncS["Street Encoder\\nLinear 2 to 64"]
    EncBu["Building Encoder\\nLinear 5 to 64"]
    EncBs["Bus Stop Encoder\\nLinear 2 to 64"]
    
    Bridge --> EncB
    Street --> EncS
    Building --> EncBu
    BusStop --> EncBs
    
    GATConv1["HeteroConv Layer 1\\nGATConv/SAGEConv\\naggr=sum"]
    Activ1["ReLU + Dropout"]
    
    EncB --> GATConv1
    EncS --> GATConv1
    EncBu --> GATConv1
    EncBs --> GATConv1
    
    GATConv1 --> Activ1
    
    GATConv2["HeteroConv Layer 2\\nGATConv/SAGEConv\\nhidden=64"]
    Activ2["ReLU"]
    
    Activ1 --> GATConv2
    GATConv2 --> Activ2
    
    FC1["Linear 64 to 32"]
    ReLU2["ReLU"]
    Drop2["Dropout"]
    FC2["Linear 32 to 1"]
    
    Activ2 --> FC1
    FC1 --> ReLU2
    ReLU2 --> Drop2
    Drop2 --> FC2
    
    Result["Output\\nBridge Importance\\n0-100"]
    FC2 --> Result
    
    style Bridge fill:#e3f2fd
    style EncB fill:#e8f5e9
    style GATConv1 fill:#fff9c4
    style GATConv2 fill:#ffe0b2
    style FC1 fill:#ffcdd2
    style Result fill:#f3e5f5
```

#### モデルの詳細解説

##### 1️⃣ 入力層：HeteroData

異種グラフの各ノードタイプは、独自の特徴量を持ちます：

| ノードタイプ | ノード数 | 特徴量次元 | 主な特徴量 |
|------------|---------|----------|-----------|
| **Bridge** | 791 | 25 | 健全度、橋齢、構造属性、環境リスク、媒介中心性 |
| **Street** | ~18,000 | 2 | 正規化座標 (x, y) |
| **Building** | ~1,000 | 5 | カテゴリ one-hot (residential/hospital/school/public/other) |
| **Bus Stop** | ~100 | 2 | プレースホルダー（将来拡張用） |

**エッジタイプ**:
- `(bridge, to, street)`: 橋梁と道路の接続
- `(bridge, to, building)`: 橋梁と建物の近接関係
- `(street, to, street)`: 道路ネットワーク
- など、合計5-8種類のエッジタイプ

##### 2️⃣ ノードエンコーダー

各ノードタイプの特徴量を、共通の埋め込み空間（64次元）に変換します。これにより、異なる特徴量次元を持つノードタイプ間でのメッセージパッシングが可能になります。

```python
# 例: Bridge ノードのエンコーディング
bridge_encoded = Linear(25 → 64)(bridge_features)  # [791, 25] → [791, 64]
```

##### 3️⃣ HeteroConv 層

**HeteroConv** は、異種グラフ上でのメッセージパッシングを実現する核心的なモジュールです。

**処理フロー**:
1. **エッジタイプごとの畳み込み**: 各エッジタイプに対して、GATConv または SAGEConv を適用
   - **GATConv**: マルチヘッドアテンション機構により、隣接ノードの重要度を動的に学習
     ```
     attention_score = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
     h_i' = Σ attention_score_ij * Wh_j
     ```
   - **SAGEConv**: 隣接ノードの特徴量を平均集約
     ```
     h_i' = σ(W · CONCAT(h_i, MEAN({h_j : j ∈ N(i)})))
     ```

2. **集約**: 各ノードタイプに対して、異なるエッジタイプから来たメッセージを集約（`aggr='sum'`）

3. **活性化**: ReLU 活性化関数とドロップアウト適用

**2層構造の意義**:
- **Layer 1**: 1-hop 近傍の情報を集約（直接接続されたノード）
- **Layer 2**: 2-hop 近傍の情報を集約（間接的につながったノード）

これにより、橋梁ノードは、直接接続された道路だけでなく、その先の建物や他の橋梁の情報も学習できます。

##### 4️⃣ 出力層

Bridge ノードの埋め込み（64次元）を、重要度スコア（1次元）に変換します。

```python
# 2層の全結合ネットワーク
x = ReLU(Linear(64 → 32)(bridge_embeddings))
x = Dropout(0.2)(x)
importance_score = Linear(32 → 1)(x)
```

**回帰タスク**: 出力は 0-100 の連続値（v1.0 の媒介中心性ベーススコアを学習）

---

#### GATConv の詳細（conv_type="GAT" の場合）

Graph Attention Networks (GAT) は、アテンション機構を用いて、隣接ノードの重要度を動的に学習します。

```mermaid
graph LR
    N1["Node i\nfeature: h_i"]
    N2["Neighbor j1"]
    N3["Neighbor j2"]
    N4["Neighbor j3"]
    
    N2 --> A1["Attention\nalpha_ij1"]
    N3 --> A2["Attention\nalpha_ij2"]
    N4 --> A3["Attention\nalpha_ij3"]
    
    A1 --> Weighted["Weighted Aggregation\nh_i' = sum alpha_ij * h_j"]
    A2 --> Weighted
    A3 --> Weighted
    
    N1 --> Weighted
    
    Weighted --> Output["Updated\nNode i feature\nh_i'"]
    
    style N1 fill:#ffeb3b
    style Output fill:#4caf50
```

**マルチヘッドアテンション** (heads=4):
- 4つの独立したアテンション機構を並列実行
- 各ヘッドが異なる「注目パターン」を学習
- 最終的に4つの出力を連結または平均

**利点**:
- ノード間の重要度を動的に学習
- 異なるエッジタイプで異なるアテンションパターンを獲得
- 解釈可能性：アテンションスコアを可視化可能

---

### 学習プロセス

#### トレーニングフロー全体像

```mermaid
flowchart TD
    Start([開始]) --> LoadData["データ読み込み\\nHeteroData"]
    
    LoadData --> Split["データ分割\\nTrain:70%, Val:15%, Test:15%"]
    
    Split --> InitModel["モデル初期化\\nBridgeImportanceHGNN"]
    
    InitModel --> InitOpt["オプティマイザ\\nAdam lr=0.001"]
    
    InitOpt --> EpochStart{エポック開始}
    
    EpochStart --> Forward["Forward Pass\\nTrain mask適用"]
    
    Forward --> Loss["損失計算\\nloss = MSE"]
    
    Loss --> Backward["Backward Pass\\nloss.backward"]
    
    Backward --> Eval["Evaluation\\nVal loss計算"]
    
    Eval --> BestCheck{Val lossが過去最小?}
    
    BestCheck -->|Yes| SaveModel["ベストモデル保存\\npatience=0"]
    BestCheck -->|No| PatienceInc["patience +1"]
    
    SaveModel --> EarlyStop{patience >= 20?}
    PatienceInc --> EarlyStop
    
    EarlyStop -->|Yes| LoadBest["ベストモデルをロード"]
    EarlyStop -->|No| MaxEpoch{epoch less than 100?}
    
    MaxEpoch -->|Yes| EpochStart
    MaxEpoch -->|No| LoadBest
    
    LoadBest --> TestEval["Test評価\\nMSE, MAE, RMSE, R2"]
    
    TestEval --> Visualize["可視化\\nTraining history"]
    
    Visualize --> SaveResults["結果保存\\nCSV, PNG files"]
    
    SaveResults --> End([完了])
    
    style Start fill:#c8e6c9
    style End fill:#c8e6c9
    style Forward fill:#fff9c4
    style Loss fill:#ffccbc
    style Backward fill:#ffe0b2
    style Eval fill:#b3e5fc
    style SaveModel fill:#c5cae9
    style TestEval fill:#f8bbd0
    style Visualize fill:#d1c4e9
```

#### 詳細な学習ステップ

##### ステップ 1: Forward Pass（順伝播）

```mermaid
sequenceDiagram
    participant Data as HeteroData
    participant Encoder as Node Encoders
    participant HConv1 as HeteroConv Layer 1
    participant HConv2 as HeteroConv Layer 2
    participant Output as Output Layer
    participant Loss as Loss Function
    
    Data->>Encoder: x_dict (各ノードタイプの特徴量)
    Note over Encoder: bridge: [791, 25]<br/>street: [N, 2]<br/>...
    
    Encoder->>HConv1: encoded_x_dict (統一次元: 64)
    Note over HConv1: ReLU + Dropout<br/>エッジタイプごとにメッセージパッシング
    
    HConv1->>HConv2: hidden_x_dict
    Note over HConv2: 2-hop 近傍情報を集約<br/>ReLU 活性化
    
    HConv2->>Output: bridge_embeddings [791, 64]
    Note over Output: Train mask 適用<br/>[554, 64] (Train のみ)
    
    Output->>Loss: predictions [554, 1]
    Note over Loss: MSE(pred, true)<br/>Train橋梁のみで損失計算
    
    Loss-->>Output: loss 値
```

##### ステップ 2: Backward Pass（逆伝播）

勾配が出力層から入力層に向かって伝播し、各パラメータが更新されます：

1. **出力層の勾配計算**: `∂L/∂W_output`
2. **HeteroConv Layer 2 の勾配**: `∂L/∂W_conv2`（各エッジタイプごと）
3. **HeteroConv Layer 1 の勾配**: `∂L/∂W_conv1`
4. **ノードエンコーダーの勾配**: `∂L/∂W_encoder`（各ノードタイプごと）

**パラメータ更新**（Adam オプティマイザ）:
```
θ_t+1 = θ_t - α · m_t / (√v_t + ε)
```
- `θ`: パラメータ
- `α`: 学習率 (0.001)
- `m_t`: 1次モーメント（勾配の移動平均）
- `v_t`: 2次モーメント（勾配の二乗の移動平均）

##### ステップ 3: Early Stopping

```mermaid
flowchart TB
    Val["Validation Loss\n計算"]
    Compare{"現在の Val Loss\nless than\n過去最小?"}
    
    Val --> Compare
    
    Compare -->|Yes| Update["ベスト更新\npatience = 0\nモデル保存"]
    Compare -->|No| Increment["patience += 1"]
    
    Increment --> Check{"patience >= 20?"}
    Update --> Continue["学習継続"]
    
    Check -->|Yes| Stop["学習停止\nベストモデルをロード"]
    Check -->|No| Continue
    
    style Update fill:#c8e6c9
    style Stop fill:#ffcdd2
    style Continue fill:#fff9c4
```

**過学習を防ぐ**: Validation loss が改善しなくなった時点で学習を停止し、最良のモデルを使用します。

##### ステップ 4: Test 評価

学習完了後、未見の Test データで最終評価を実施：

| 評価指標 | 説明 | 計算式 |
|---------|------|--------|
| **MSE** | 平均二乗誤差 | `Σ(y_pred - y_true)² / n` |
| **MAE** | 平均絶対誤差 | `Σ|y_pred - y_true| / n` |
| **RMSE** | 二乗平均平方根誤差 | `√MSE` |
| **R²** | 決定係数 | `1 - Σ(y_pred - y_true)² / Σ(y_true - ȳ)²` |

**R² の解釈**:
- R² = 1.0: 完璧な予測
- R² = 0.8: モデルが分散の80%を説明
- R² = 0.0: 平均値予測と同等
- R² < 0.0: 平均値より悪い予測

---

### 実装上の工夫

#### 1. Heterogeneous Graph の特徴

従来の同種グラフ（Homogeneous Graph）と異なり、異種グラフは以下の特徴を持ちます：

```mermaid
graph TD
    subgraph Homo["同種グラフ 従来"]
        H1[ノード] --- H2[ノード]
        H2 --- H3[ノード]
        H3 --- H4[ノード]
    end
    
    subgraph Hetero["異種グラフ 本研究"]
        B1[Bridge] --- S1[Street]
        B1 --- Bu1[Building]
        S1 --- S2[Street]
        Bu1 --- S2
        B2[Bridge] --- S2
        S2 --- Bs1[Bus Stop]
    end
    
    style H1 fill:#90caf9
    style H2 fill:#90caf9
    style H3 fill:#90caf9
    style H4 fill:#90caf9
    style B1 fill:#ef5350
    style B2 fill:#ef5350
    style S1 fill:#66bb6a
    style S2 fill:#66bb6a
    style Bu1 fill:#ffa726
    style Bs1 fill:#ab47bc
```

**異種グラフの利点**:
- 各ノードタイプが専用の特徴量を持てる
- エッジタイプごとに異なる関係性をモデル化
- より現実世界の複雑な構造を表現可能

#### 2. メッセージパッシングの可視化

例：ある橋梁ノード `BR_0530`（最高スコア橋梁）のメッセージパッシング

```mermaid
graph TB
    BR["BR_0530 流通ICオンランプB橋"] -->|to_street| S1["Street_1234"]
    BR -->|to_street| S2["Street_5678"]
    BR -->|to_building| B1["Building_A 病院"]
    BR -->|to_building| B2["Building_B 学校"]
    
    S1 -->|street_to_street| S3["Street_9012"]
    S2 -->|street_to_street| S4["Street_3456"]
    S3 --> BR2["BR_1234 隣接橋梁"]
    B1 --> S5["Street_7890"]
    
    style BR fill:#ef5350,color:#fff
    style BR2 fill:#ef5350,color:#fff
```

**情報の流れ**:
1. **Layer 1**: `BR_0530` は直接接続された道路・建物から情報を受け取る
2. **Layer 2**: さらにその先の道路網や他の橋梁の情報も間接的に学習
3. **最終埋め込み**: 局所的＋広域的な文脈を統合した表現を獲得

#### 3. Train/Val/Test 分割の重要性

```mermaid
pie title データ分割 791橋梁
    "Train 70percent" : 554
    "Validation 15percent" : 119
    "Test 15percent" : 118
```

- **Train**: モデルのパラメータ学習に使用
- **Validation**: ハイパーパラメータ調整と Early stopping に使用
- **Test**: 最終評価のみ使用（学習中は一切使わない）

**リーク防止**: Test データは学習プロセスから完全に隔離され、真の汎化性能を評価



---

---

## 📊 期待される結果

v1.0 の NetworkX 媒介中心性ベースのスコアを ground truth として学習した場合、以下のような性能が期待されます：

### 評価指標の目標値

```mermaid
graph LR
    R2["評価指標 R2 >= 0.7-0.9"]
    MAE["評価指標 MAE <= 5-10"]
    RMSE["評価指標 RMSE <= 8-12"]
    
    style R2 fill:#c8e6c9
    style MAE fill:#fff9c4
    style RMSE fill:#ffccbc
```

### 予測精度の解釈

#### R² スコアの意味

```mermaid
graph LR
    E1["R2 = 1.0 完璧"]
    E2["R2 = 0.8-0.9 優秀"]
    E3["R2 = 0.6-0.8 良好"]
    E4["R2 = 0.4-0.6 普通"]
    E5["R2 less than 0.4 改善必要"]
    
    E1 --> E2 --> E3 --> E4 --> E5
    
    style E1 fill:#4caf50,color:#fff
    style E2 fill:#8bc34a,color:#fff
    style E3 fill:#ffeb3b
    style E4 fill:#ff9800,color:#fff
    style E5 fill:#f44336,color:#fff
```

**R² = 0.8 の例**:
- 重要度スコアの分散の80%をモデルが説明
- 残り20%は、モデルが捉えていない要因（ノイズ、未観測変数など）

#### 予測誤差の分布（理想的なケース）

```mermaid
graph TD
    Center["中心 誤差ゼロに近い"]
    Left["左裾 過小予測"]
    Right["右裾 過大予測"]
    
    Left --> Center
    Center --> Right
    
    style Center fill:#4caf50,color:#fff
    style Left fill:#ff9800,color:#fff
    style Right fill:#ff9800,color:#fff
```

理想的には、予測誤差が正規分布に従い、大きな誤差が少ない状態を目指します。

### 実際の出力例

学習完了後、以下のような可視化が生成されます：

#### 1. Training History（学習履歴）

```
Epoch  Train Loss  Val Loss  Train MAE  Val MAE
  10      45.23      48.91      5.12      5.45
  20      28.67      32.14      3.89      4.21
  30      18.45      22.38      3.12      3.67
  40      12.89      18.76      2.45      3.22
  50      10.23      17.91      2.11      3.05  ← Best Val Loss
  60      9.87       18.45      2.03      3.18
  70      9.12       19.23      1.89      3.34  ← Early Stopping
```

**グラフの見方**:
- Train Loss が順調に減少 → 学習が進行
- Val Loss が途中から上昇 → 過学習の兆候
- Early Stopping が適切に動作 → 過学習前でストップ

#### 2. Predictions vs Truth（予測 vs 真値）

理想的な散布図のパターン：

```mermaid
graph LR
    Good["優良 R2>=0.8"]
    Medium["中程度 R2=0.6-0.8"]
    Poor["改善必要 R2 less than 0.6"]
    
    Good --> Medium --> Poor
    
    style Good fill:#4caf50,color:#fff
    style Medium fill:#ffeb3b
    style Poor fill:#f44336,color:#fff
```

### スコア範囲別の予測精度

HGNN は、スコア範囲によって予測精度が異なる可能性があります：

| スコア範囲 | 予測難易度 | 理由 |
|-----------|----------|------|
| **0-20 (Very Low)** | 🟢 易しい | データ数が多い（659橋、83%）|
| **20-40 (Low)** | 🟢 易しい | データ数が多い（85橋、11%）|
| **40-70 (Medium-High)** | 🟡 中程度 | データ数が中程度（41橋、5%）|
| **70-100 (Critical)** | 🔴 難しい | データ数が少ない（6橋、0.8%）、**不均衡データ問題**|

**対策**:
- クラス重み付け損失関数
- SMOTE（Synthetic Minority Over-sampling Technique）
- Focal Loss の適用

**注意**: これらは参考値であり、実際のデータや設定により異なります。学習結果は `output/hgnn_training/test_metrics.csv` で確認できます。

---

## 🚀 将来の拡張

v1.1 は HGNN の基盤を提供します。今後の拡張可能性：

### 開発ロードマップ

```mermaid
gantt
    title Bridge Importance HGNN ロードマップ
    dateFormat YYYY-MM-DD
    
    section v1.0 完了
    NetworkX基盤, 媒介中心性, 異種グラフ構築 :done, v10, 2024-01-01, 2024-12-31
    
    section v1.1 現在
    PyTorch Geo統合, HeteroConv, 特徴量 :active, v11, 2025-01-01, 2025-12-31
    
    section v1.2 計画中
    特徴量拡張, 交通量, 人口密度, 災害リスク :v12, 2026-01-01, 2026-12-31
    
    section v2.0 将来
    Temporal GNN, Multi-task, Attention可視化 :v20, 2027-01-01, 2027-12-31
    
    section v3.0 長期
    実用化, API, Webアプリ, 自治体統合 :v30, 2028-01-01, 2028-12-31
```

### 拡張の詳細

#### 1. 新しい特徴量の追加

```mermaid
graph TD
    CF1["v1.1\n橋梁: 健全度, 橋齢, 構造"]
    CF2["v1.1\n道路: 座標"]
    CF3["v1.1\n建物: カテゴリ"]
    
    Traffic["交通量データ\n実測AADT値, 時間帯別"]
    Population["人口密度\nメッシュ別, 昼夜間"]
    Disaster["災害リスク\n地震, 洪水, 津波"]
    Temporal["時系列データ\n点検履歴, 劣化進行"]
    Economic["経済データ\n産業集積, 地価"]
    
    CF1 --> Traffic
    CF1 --> Population
    CF1 --> Disaster
    CF2 --> Temporal
    CF3 --> Economic
    
    style Traffic fill:#ffeb3b
    style Population fill:#4caf50
    style Disaster fill:#f44336,color:#fff
    style Temporal fill:#2196f3,color:#fff
    style Economic fill:#9c27b0,color:#fff
```

**期待される効果**:
- 予測精度の向上（R² > 0.9）
- より現実的な重要度評価
- 災害時の影響予測

#### 2. モデルの改良

```mermaid
graph LR
    V11["v1.1\nStatic HGNN\n単一タスク"]
    V12["v1.2\nDynamic HGNN\n時系列対応"]
    V20["v2.0\nTemporal GNN\nMulti-task"]
    V30["v3.0\nRL最適化"]
    
    V11 --> V12 --> V20 --> V30
    
    MTL["Multi-Task Learning\n同時学習"]
    TGNN["Temporal GNN\n時系列モデリング"]
    Attention["Attention可視化\n解釈性向上"]
    WhatIf["What-if分析\nシミュレーション"]
    
    V12 --> MTL
    V20 --> TGNN
    V20 --> Attention
    V30 --> WhatIf
    
    style V11 fill:#90caf9
    style V12 fill:#66bb6a
    style V20 fill:#ffa726
    style V30 fill:#ab47bc,color:#fff
```

##### Multi-Task Learning の例

```mermaid
graph TB
    Input["Shared HeteroConv Layers\n共通特徴抽出"]
    
    Task1["Task 1: 重要度予測\n出力: 0-100スコア, 損失: MSE"]
    Task2["Task 2: 健全度予測\n出力: I/II/III/IV, 損失: CE"]
    Task3["Task 3: 劣化速度予測\n出力: mm/年, 損失: MSE"]
    
    Input --> Task1
    Input --> Task2
    Input --> Task3
    
    Total["Total Loss\nL = a*L1 + b*L2 + c*L3"]
    
    Task1 --> Total
    Task2 --> Total
    Task3 --> Total
    
    style Input fill:#90caf9
    style Task1 fill:#66bb6a
    style Task2 fill:#ffa726
    style Task3 fill:#ab47bc,color:#fff
    style Total fill:#ef5350,color:#fff
```

**利点**:
- 複数タスク間で知識を共有
- データ不足タスクの精度向上
- より一般化された特徴表現

#### 3. アプリケーション展開

```mermaid
graph TD
    API["REST API\nリアルタイム予測, バッチ処理"]
    WebApp["Web アプリ\n地図, シミュレータ, レポート"]
    Mobile["モバイルアプリ\n現地点検, AR可視化"]
    Integration["自治体統合\nシステム連携, アラート"]
    Optimization["最適化エンジン\n補修計画, 予算配分"]
    
    API --> WebApp
    API --> Mobile
    WebApp --> Integration
    Mobile --> Integration
    API --> Optimization
    
    style API fill:#2196f3,color:#fff
    style WebApp fill:#4caf50,color:#fff
    style Mobile fill:#9c27b0,color:#fff
    style Integration fill:#ff9800,color:#fff
    style Optimization fill:#f44336,color:#fff
```

#### 実用化のユースケース

```mermaid
sequenceDiagram
    participant User as 自治体担当者
    participant Web as Webアプリ
    participant API as 予測API
    participant DB as データベース
    participant Model as HGNNモデル
    
    User->>Web: 橋梁選択
    Web->>API: What-ifリクエスト BR_0123 repair
    API->>DB: 現在のデータ取得
    DB-->>API: 橋梁グラフデータ
    API->>Model: シミュレーション実行 健全度III to I
    Model-->>API: 予測結果: 重要度変化, 周辺影響
    API-->>Web: レスポンス返却 JSON
    Web-->>User: 結果表示: Before/After 地図
    
    Note over User,Model: 意思決定支援サイクル
```

### 技術的課題と対策

| 課題 | 対策 |
|------|------|
| **スケーラビリティ** | GraphSAINT サンプリング、ミニバッチ学習 |
| **リアルタイム性** | モデル軽量化、キャッシング、推論最適化 |
| **データ不足** | Transfer Learning、Data Augmentation、合成データ |
| **解釈可能性** | Attention 可視化、SHAP 値、Layer-wise Relevance Propagation |
| **モデル更新** | Incremental Learning、Online Learning、A/B Testing |



---

## 🐛 既知の制限事項

- **データ依存性**: 学習は v1.0 の媒介中心性スコアを ground truth として使用するため、その精度に依存します
- **計算リソース**: GPU がない環境ではトレーニングに時間がかかる場合があります（CPU でも動作可能）
- **OSMnx API**: v1.0 から継承された OSM building/POI 取得の不安定性が残っています

---

## 📝 変更点まとめ

### 新規ファイル

- `hetero_data_converter.py`
- `hgnn_model.py`
- `train_hgnn.py`
- `convert_to_heterodata.py`
- `RELEASE_NOTES_v1.1.0.md`

### 更新ファイル

- `requirements.txt` - PyTorch Geometric 依存関係追加
- `config.yaml` - `hgnn:` セクション追加
- `VERSION` - 1.0.0 → 1.1.0
- `CHANGELOG.md` - v1.1.0 エントリ追加
- `README.md` - v1.1 使用方法追加
- `README_JP.md` - v1.1 機能説明追加
- `main.py` - バージョン 1.1.0 に更新
- `run_visualization.py` - バージョン 1.1.0 に更新

---

## 👥 貢献者

- v1.1 HGNN 統合: GitHub Copilot + User
- v1.0 基盤実装: Project Team

---

## 📞 サポート

質問や問題がある場合は、GitHub Issues を利用してください。

**Happy Bridge Importance Prediction! 🌉🤖**
