import sys
from pathlib import Path
from shutil import copytree, ignore_patterns


# このスクリプトは新しいPyTorchプロジェクトをテンプレートファイルを用いて初期化します。
# `python3 new_project.py ../MyNewProject` と実行すると、MyNewProjectという名前の新しいプロジェクトが作成されます。

current_dir = Path() # 現在のディレクトリのパスを取得

# スクリプトがpytorch-templateディレクトリ内で実行されているかを確認
assert (current_dir / 'new_project.py').is_file(), 'Script should be executed in the pytorch-template directory'
# コマンドライン引数の数を確認（プロジェクト名を指定する必要がある）
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: python3 new_project.py MyNewProject'

# コマンドラインから受け取ったプロジェクト名をPathオブジェクトに変換
project_name = Path(sys.argv[1])
# 新しいプロジェクトのディレクトリのパスを生成
target_dir = current_dir / project_name

# コピーから除外するファイルおよびディレクトリのリスト
ignore = [".git", "data", "saved", "new_project.py", "LICENSE", ".flake8", "README.md", "__pycache__"]
# 現在のディレクトリから新しいプロジェクトディレクトリにファイルをコピー
copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
# 新しいプロジェクトの初期化が完了したことをユーザーに通知
print('New project initialized at', target_dir.absolute().resolve())