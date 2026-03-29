"""
バス停取得のテストスクリプト
"""
import osmnx as ox
import geopandas as gpd
import yaml
from shapely.geometry import box
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設定読み込み
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# テスト範囲（山口市中心部の小さなエリア）
test_bbox = {
    'min_lon': 131.45,
    'min_lat': 34.15,
    'max_lon': 131.50,
    'max_lat': 34.20
}

logger.info(f"Testing bus stop fetch in area: {test_bbox}")

# 境界ポリゴン作成
polygon = box(
    test_bbox['min_lon'],
    test_bbox['min_lat'],
    test_bbox['max_lon'],
    test_bbox['max_lat']
)

# バス停取得を試行
logger.info("Attempting to fetch bus stops with highway='bus_stop'...")
try:
    tags = {'highway': 'bus_stop'}
    bus_stops = ox.features_from_polygon(polygon, tags=tags)
    logger.info(f"✓ Found {len(bus_stops)} bus stops with highway='bus_stop'")
    if len(bus_stops) > 0:
        print("\n=== Sample bus stops ===")
        print(bus_stops[['name', 'geometry']].head())
except Exception as e:
    logger.error(f"✗ Failed with highway='bus_stop': {e}")

# 代替タグで試行
logger.info("\nAttempting to fetch bus stops with amenity='bus_station'...")
try:
    tags = {'amenity': 'bus_station'}
    bus_stations = ox.features_from_polygon(polygon, tags=tags)
    logger.info(f"✓ Found {len(bus_stations)} bus stations with amenity='bus_station'")
    if len(bus_stations) > 0:
        print("\n=== Sample bus stations ===")
        print(bus_stations[['name', 'geometry']].head())
except Exception as e:
    logger.error(f"✗ Failed with amenity='bus_station': {e}")

# 公共交通全般で試行
logger.info("\nAttempting to fetch all public transport...")
try:
    tags = {'public_transport': True}
    public_transport = ox.features_from_polygon(polygon, tags=tags)
    logger.info(f"✓ Found {len(public_transport)} public_transport features")
    if len(public_transport) > 0:
        print("\n=== Sample public transport ===")
        print(public_transport[['public_transport', 'name', 'geometry']].head(10))
        
        # public_transportのタイプ別集計
        if 'public_transport' in public_transport.columns:
            print("\n=== public_transport types ===")
            print(public_transport['public_transport'].value_counts())
except Exception as e:
    logger.error(f"✗ Failed with public_transport: {e}")

logger.info("\n=== Test completed ===")
