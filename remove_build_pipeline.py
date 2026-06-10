import re
from pathlib import Path

content = Path("src/knox/cli/commands.py").read_text(encoding="utf-8")
content = re.sub(r'def _build_pipeline_config\([\s\S]*?^def ', 'def ', content, flags=re.MULTILINE)
Path("src/knox/cli/commands.py").write_text(content, encoding="utf-8")