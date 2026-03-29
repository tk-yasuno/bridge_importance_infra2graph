"""
セットアップと実行サポートスクリプト
Bridge Importance Scoring MVP
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Pythonバージョンチェック"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")


def install_dependencies():
    """依存パッケージのインストール"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def check_data_files():
    """データファイルの存在確認"""
    print("\nChecking data files...")
    
    data_files = {
        'Bridge data': Path('data/Bridge_xy_location/YamaguchiPrefBridgeListOpen251122_154891.xlsx'),
        'River data': Path('data/RiverDataKokudo/W05-08_35_GML'),
        'Coastline data': Path('data/KaigansenDataKokudo/C23-06_35_GML')
    }
    
    all_exist = True
    for name, path in data_files.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name} NOT FOUND: {path}")
            all_exist = False
    
    return all_exist


def create_output_directory():
    """出力ディレクトリの作成"""
    output_dir = Path('output/bridge_importance')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory created: {output_dir}")


def run_main_pipeline():
    """メインパイプラインの実行"""
    print("\n" + "=" * 70)
    print("Starting Bridge Importance Scoring Pipeline")
    print("=" * 70 + "\n")
    
    try:
        import main
        main.main()
        print("\n" + "=" * 70)
        print("Pipeline completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """セットアップと実行"""
    print("=" * 70)
    print("Bridge Importance Scoring MVP - Setup & Run")
    print("=" * 70)
    
    # 1. Pythonバージョンチェック
    check_python_version()
    
    # 2. データファイル確認
    data_ok = check_data_files()
    if not data_ok:
        print("\n⚠ Warning: Some data files are missing.")
        print("Please ensure all required data files are in place.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            sys.exit(1)
    
    # 3. 依存パッケージのインストール（オプション）
    response = input("\nInstall/update dependencies? (y/n): ")
    if response.lower() == 'y':
        install_dependencies()
    
    # 4. 出力ディレクトリ作成
    create_output_directory()
    
    # 5. パイプライン実行確認
    print("\n" + "=" * 70)
    response = input("Run the main pipeline now? (y/n): ")
    if response.lower() == 'y':
        run_main_pipeline()
    else:
        print("\nSetup complete. Run 'python main.py' to start the pipeline.")


if __name__ == '__main__':
    main()
