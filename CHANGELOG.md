# Changelog

All notable changes to the Bridge Importance Scoring MVP project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-29

### Added - HGNN Integration
- **PyTorch Geometric Integration**
  - `hetero_data_converter.py`: NetworkX to HeteroData conversion module
  - `hgnn_model.py`: Heterogeneous GNN model definitions (HeteroConv + GATConv/SAGEConv)
  - `train_hgnn.py`: Training and evaluation pipeline for HGNN
  - `convert_to_heterodata.py`: Data conversion execution script

- **Node Feature Engineering**
  - **Bridge Nodes (25 features)**:
    - Health condition (one-hot encoded, 4 classes: 健全度Ⅰ-Ⅳ)
    - Bridge age, length, width
    - Environmental risks: log(distance to river), log(distance to coast)
    - Existing centrality metrics: betweenness
    - Nearby facility counts: buildings, hospitals, schools, public facilities
    - Binary flags: 離島架橋, 長大橋, 特殊橋, 重要物流道路, 緊急輸送道路, 跨線橋, 跨道橋
  - **Street Nodes**: Normalized x/y coordinates
  - **Building Nodes**: One-hot encoded categories (residential, hospital, school, public, other)
  - **Bus Stop Nodes**: Placeholder features (for future expansion)

- **HGNN Model Architecture**
  - Standard model: Multi-layer HeteroConv (2 layers default)
  - Simple model: Single-layer HeteroConv for small-scale data
  - GAT (Graph Attention Networks) or GraphSAGE convolution options
  - Configurable: hidden channels (64), dropout (0.2), attention heads (4)

- **Training Infrastructure**
  - Train/validation/test split (70%/15%/15%)
  - Early stopping with patience (20 epochs default)
  - MSE loss for regression
  - Evaluation metrics: MSE, MAE, RMSE, R²
  - Training history visualization (loss/MAE curves)
  - Prediction vs. ground truth scatter plots

- **Configuration**
  - New `hgnn:` section in `config.yaml`
  - Hyperparameter settings: learning rate, num_epochs, weight_decay, dropout
  - Model architecture options: model_type, conv_type, hidden_channels, num_layers

### Changed
- **requirements.txt**: Enabled PyTorch and PyTorch Geometric dependencies
  - `torch>=2.0.0`
  - `torch-geometric>=2.3.0`
  - `torch-scatter>=2.1.0`
  - `torch-sparse>=0.6.17`
  - `scikit-learn>=1.0.0`

- **README.md**: Updated to v1.1.0 with HGNN usage instructions

### Technical Details
- **Target Task**: Node regression on bridge nodes (predict importance_score 0-100)
- **Graph Structure**: Heterogeneous graph with 4 node types (bridge, street, building, bus_stop) and multiple edge types
- **Training Device**: Auto-detection (CUDA if available, else CPU)
- **Model Output**: Bridge importance score predictions with evaluation on held-out test set

## [1.0.0] - 2026-03-29

### Added
- **Core Pipeline Implementation**
  - Main pipeline orchestration (`main.py`)
  - Data loading module with city filtering (`data_loader.py`)
  - Heterogeneous graph construction with OSMnx integration (`graph_builder.py`)
  - Betweenness centrality computation and scoring (`centrality_scorer.py`)
  - Narrative generation for explainability (`narrative_generator.py`)
  - Utility functions and validation (`utils.py`)

- **Visualization System**
  - Score distribution analysis (4-panel statistical charts)
  - Geographic visualization of top bridges
  - Interactive web map using Folium with rich tooltips
  - Visualization runner script (`run_visualization.py`)

- **Configuration System**
  - YAML-based configuration (`config.yaml`)
  - City filtering (Yamaguchi City: 791 bridges)
  - Adjustable proximity thresholds
  - Configurable scoring weights
  - Buffer size configuration (1.0 km)

- **Data Processing**
  - Excel bridge data loader with encoding handling
  - Shapefile integration (rivers, coastlines)
  - OSM street network fetching with timeout handling (300s)
  - Bridge-to-road network snapping (30m threshold)
  - River and coastline proximity computation

- **Scoring System**
  - Betweenness centrality as primary metric (60% weight)
  - Public facility access score (20% weight)
  - Traffic proxy score (20% weight)
  - 0-100 normalized importance score
  - 5-tier categorization (critical/high/medium/low/very_low)

- **Output Formats**
  - CSV export with full bridge attributes
  - GeoJSON for mapping applications
  - Markdown detailed report
  - Top 10 critical bridges CSV
  - Interactive HTML map
  - Graph object serialization (Pickle)
  - Execution metadata (YAML)
  - Statistical distribution charts (PNG)
  - Geographic maps (PNG)

- **Documentation**
  - Comprehensive README.md with algorithm flow diagram
  - Japanese documentation (README_JP.md)
  - Quick start guide (QUICKSTART.md)
  - Lesson learned document (Lesson_Yamaguchi_City_Bridge_Importance.md)
  - Visualization results with detailed analysis
  - Setup and run script documentation

- **Error Handling & Robustness**
  - OSMnx API error handling with fallback
  - Timeout configuration for large-area OSM fetches
  - Data validation and column name verification
  - Missing data handling
  - CRS detection and conversion
  - Logging system with file and console output

### Results Summary (Yamaguchi City)
- **Processed**: 791 bridges (filtered from 4,293 county-wide)
- **Processing Area**: 1,948.1 km² (reduced from 18,720.5 km²)
- **Graph Size**: 18,998 nodes, 25,666 edges
- **Execution Time**: 36 minutes 28 seconds
  - Data loading: 4s
  - Graph construction: 56s
  - Centrality computation: 35m 22s (97% of total time)
  - Narrative generation & output: 6s

- **Top Findings**:
  - 6 high-importance bridges (0.8%)
  - 41 medium-importance bridges (5.2%)
  - Geographic concentration in Ogori district
  - 52.2% of bridges within 50m of rivers (flood risk)
  - Historic infrastructure: Aoi Bridge (1931, 93 years old) ranks 7th

### Fixed Issues
- **Issue #1**: OSM data fetch timeout on large areas
  - Solution: Reduced buffer from 5km to 1km
  - Solution: Added city filtering to focus on Yamaguchi City only
  - Solution: Implemented OSMnx timeout setting (300s)

- **Issue #2**: City column name mismatch
  - Solution: Verified actual column structure in Excel
  - Solution: Changed from "市町村" to "所在地"

- **Issue #3**: OSMnx API changes
  - Solution: Implemented error handling for missing `geometries_from_polygon`
  - Solution: Graceful degradation (continue without building/bus data)

### Performance Optimizations
- City-level filtering: 81.6% bridge count reduction
- Spatial filtering: 89.6% area reduction
- OSM cache enabled for repeated runs
- Bridge-to-street snapping: 97% success rate (767/791)

### Known Limitations
- Betweenness centrality computation is O(n³), limiting scalability
- Building and bus stop data currently unavailable due to OSMnx API changes
- Approximate 36-minute runtime for ~19k node graphs
- No GPU acceleration implemented

### Dependencies
- Python 3.8+
- NetworkX 2.6+
- GeoPandas 0.10+
- OSMnx 1.1+
- Folium 0.12+
- Matplotlib 3.4+
- Pandas 1.3+
- PyYAML 5.4+

---

## Future Roadmap

### v1.1 (Planned)
- Fix OSMnx API for building/bus stop data
- Add sampling-based centrality approximation for large graphs
- Implement parallel processing for centrality computation
- Add more visualization types (heatmaps, network diagrams)

### v2.0 (Planned)
- PyTorch Geometric integration for HGNN
- Machine learning-based importance prediction
- Temporal analysis with historical data
- Real-time traffic data integration
- Scenario simulation (bridge closure impact)

---

[1.0.0]: https://github.com/yourusername/bridge_importance_score/releases/tag/v1.0.0
