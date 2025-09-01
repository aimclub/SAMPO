"""Generate a README with docstrings of SAMPO package. / Генерирует README с docstring'ами пакета SAMPO."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def collect_docstrings(package_path: Path) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
    """Collect docstrings from package modules. / Собирает docstring'и из модулей пакета."""

    data: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"classes": {}, "functions": {}})
    )
    for py_file in package_path.rglob("*.py"):
        rel_dir = py_file.parent.relative_to(package_path)
        dir_key = str(rel_dir) if str(rel_dir) != "." else "root"
        source = py_file.read_text(encoding="utf-8")
        module_ast = ast.parse(source)
        for node in module_ast.body:
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                data[dir_key][py_file.name]["classes"][node.name] = doc
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                data[dir_key][py_file.name]["functions"][node.name] = doc
    return data


def format_section(name: str, items: Dict[str, str]) -> List[str]:
    """Format docstrings section. / Форматирует секцию docstring'ов."""

    lines: List[str] = []
    if items:
        header = "Classes / Классы" if name == "classes" else "Functions / Функции"
        lines.append(f"#### {header}")
        for obj, doc in sorted(items.items()):
            lines.append(f"- **{obj}**")
            if doc:
                doc_lines = [f"  {line}" for line in doc.strip().splitlines()]
                lines.extend(doc_lines)
            lines.append("")
    return lines


def build_anchor(parts: List[str]) -> str:
    """Create an anchor id from path parts. / Создаёт идентификатор якоря из частей пути."""

    slug = "-".join(parts)
    slug = re.sub(r"[^a-zA-Z0-9_-]", "", slug)
    return slug.lower()


def generate_readme(package_path: Path, output_path: Path) -> None:
    """Create README.md summarizing docstrings. / Создаёт README.md с перечнем docstring'ов."""

    data = collect_docstrings(package_path)
    lines: List[str] = [
        "# Docstring Reference / Справочник docstring'ов",
        "",
        "Automatically collected docstrings from the SAMPO package. / Автоматически собранные docstring'и из пакета SAMPO.",
        "",
    ]

    # Prepare table of contents. / Подготавливаем оглавление.
    toc_lines: List[str] = ["## Table of Contents / Оглавление", ""]

    for dir_key in sorted(data.keys()):
        dir_anchor = build_anchor([dir_key])
        toc_lines.append(f"- [{dir_key}](#{dir_anchor})")
        for file_name in sorted(data[dir_key].keys()):
            file_anchor = build_anchor([dir_key, file_name])
            toc_lines.append(f"  - [{file_name}](#{file_anchor})")

    lines.extend(toc_lines)
    lines.append("")

    for dir_key in sorted(data.keys()):
        dir_anchor = build_anchor([dir_key])
        lines.append(f"## <a id=\"{dir_anchor}\"></a>{dir_key}")
        lines.append("")
        for file_name in sorted(data[dir_key].keys()):
            file_anchor = build_anchor([dir_key, file_name])
            rel_path = (
                Path(dir_key) / file_name
                if dir_key != "root"
                else Path(file_name)
            )
            # Link to the actual file. / Ссылка на сам файл.
            lines.append(
                "### "
                f"<a id=\"{file_anchor}\"></a>"
                f"[{file_name}]({rel_path.as_posix()})"
            )
            lines.append("")
            sections = data[dir_key][file_name]
            for sect in ("classes", "functions"):
                lines.extend(format_section(sect, sections[sect]))
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    package_root = Path(__file__).parent
    generate_readme(package_root, package_root / "README.md")
