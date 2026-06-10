import re
from pathlib import Path

content = Path("tests/unit/test_throughput_features.py").read_text(encoding="utf-8")

content = content.replace("verify_node(node, tmp_path)", "verify_node(node, tmp_path, model='test/model')")
content = content.replace("verify_node(node, sources)", "verify_node(node, sources, model='test/model')")
content = content.replace("verify_wiki(wiki, tmp_path)", "verify_wiki(wiki, tmp_path, model='test/model')")

Path("tests/unit/test_throughput_features.py").write_text(content, encoding="utf-8")