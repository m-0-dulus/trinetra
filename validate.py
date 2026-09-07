import ast
from pathlib import Path
r=Path(__file__).parent
for p in r.rglob("*.py"): ast.parse(p.read_text())
for p in ["server.py","requirements.txt","phone/index.html","phone/app.js","phone/style.css","dashboard/index.html","dashboard/app.js","dashboard/style.css","ai/target_manager.py"]:
    assert (r/p).exists(), p
print("VALIDATION PASS")
