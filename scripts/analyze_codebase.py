"""Analyze codebase line counts for refactoring planning."""

import sys
from pathlib import Path


def count_lines(filepath: str) -> tuple[int, int, int, int, int]:
    """Count lines: total, imports, blanks, comments, net code."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    total = len(lines)
    blanks = 0
    comments = 0
    imports = 0
    net = 0
    in_multiline = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blanks += 1
            continue
        if in_multiline:
            if '"""' in stripped or "'''" in stripped:
                in_multiline = False
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            comments += 1
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports += 1
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_multiline = not (stripped.count('"""') >= 2 or stripped.count("'''") >= 2)
            continue
        net += 1

    return total, imports, blanks, comments, net


def walk_src(base: str) -> dict:
    """Walk Python files under src/knox, return results grouped by directory."""
    src_dir = Path(base) / "src" / "knox"
    results: dict[str, dict] = {}

    for pyfile in sorted(src_dir.rglob("*.py")):
        rel = pyfile.relative_to(src_dir)
        parent = str(rel.parent)
        if parent == ".":
            parent = ""

        try:
            counts = count_lines(pyfile)
        except Exception:
            counts = (0, 0, 0, 0, 0)

        results[str(rel)] = {
            "path": str(rel),
            "dir": parent,
            "total": counts[0],
            "imports": counts[1],
            "blanks": counts[2],
            "comments": counts[3],
            "net": counts[4],
        }

    return results


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    results = walk_src(base)

    # Group by directory
    dirs: dict[str, dict] = {}
    for name, info in results.items():
        d = info["dir"]
        if d not in dirs:
            dirs[d] = {"files": [], "total": 0, "net": 0, "comments": 0, "blanks": 0}
        dirs[d]["files"].append(info)
        dirs[d]["total"] += info["total"]
        dirs[d]["net"] += info["net"]
        dirs[d]["comments"] += info["comments"]
        dirs[d]["blanks"] += info["blanks"]

    # Sort directories by net lines descending
    sorted_dirs = sorted(dirs.keys(), key=lambda x: dirs[x]["net"], reverse=True)

    print(f"{'Directory':<37} {'Files':>5} {'Total':>6} {'Net':>6} {'Cmt':>6} {'Blk':>6}")
    print("-" * 70)
    all_over = []
    for d in sorted_dirs:
        info = dirs[d]
        nfiles = len(info["files"])
        n_over350 = sum(1 for f in info["files"] if f["net"] > 350)
        n_over = sum(1 for f in info["files"] if f["net"] > 200)
        flag = ""
        if n_over350:
            flag = " ***"
        elif n_over:
            flag = " *"
        label = "src/knox/" + d if d else "src/knox"
        print(f"{label:<37} {nfiles:>5} {info['total']:>6} {info['net']:>6} {info['comments']:>6} {info['blanks']:>6}{flag}")

        for f in info["files"]:
            if f["net"] > 350:
                all_over.append((f["net"], f["path"], ">350"))
            elif f["net"] > 200:
                all_over.append((f["net"], f["path"], ">200"))

    # Files over 350 net lines
    print("\n\n=== HIGH PRIORITY: Files over 350 net lines ===")
    all_over.sort(key=lambda x: -x[0])
    printed = set()
    for net, path, tag in all_over:
        if tag == ">350" and path not in printed:
            print(f"  {net:>6}  {path}")
            printed.add(path)

    print("\n=== FILES over 200 net lines ===")
    printed = set()
    for net, path, tag in all_over:
        if path not in printed:
            print(f"  {net:>6}  {path}")
            printed.add(path)

    # Submodules with more than 7 files
    print("\n\n=== Submodules with 7+ files (top-level .py, excl __init__) ===")
    for d in sorted_dirs:
        info = dirs[d]
        # Normalize separators so depth works on both Windows and POSIX
        d_norm = d.replace("\\", "/")
        depth = d_norm.count("/") + 1 if d_norm else 0
        own_files = [
            f for f in info["files"]
            if not f["path"].endswith("__init__.py")
            and f["path"].replace("\\", "/").count("/") == depth
        ]
        n = len(own_files)
        if n > 7:
            label = f"src/knox/{d}/" if d else "src/knox/"
            print(f"  {n:>3} files  {label}")
            for f in sorted(own_files, key=lambda x: -x["net"]):
                print(f"         {f['net']:>6}  {f['path']}")


if __name__ == "__main__":
    main()
