#!/usr/bin/env python3

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Set

VALID_EXTENSIONS = {'.py', '.html'}
EXCLUDED_FILES = {'__init__.py', 'setup.py', 'main.py'}
EXCLUDED_FOLDERS = {'__pycache__', '.venv', 'venv', '.git', 'build', 'dist', 'docs', 'tests'}

STANDARD_MODULES = (
    'os.', 'sys.', 're.', 'pathlib.', 'argparse.', 'typing.', 'datetime.', 'json.',
    'logging.', 'collections.', 'time.', 'string.', 'random.', 'subprocess.'
)

def get_app_name(root_dir: str) -> str:
    return Path(root_dir).resolve().name

def load_gitignore_patterns(root_dir: str) -> Set[str]:
    patterns = set()
    gitignore = Path(root_dir) / '.gitignore'
    if gitignore.exists():
        with gitignore.open('r') as f:
            for line in f:
                pattern = line.strip()
                if pattern and not pattern.startswith('#'):
                    patterns.add(pattern)
    return patterns

def is_ignored(path: Path, root_dir: Path, patterns: Set[str]) -> bool:
    relative_path = path.relative_to(root_dir)
    if any(part in EXCLUDED_FOLDERS for part in relative_path.parts):
        return True
    if path.name in EXCLUDED_FILES:
        return True
    for pattern in patterns:
        if relative_path.match(pattern):
            return True
    return False

def get_folder_structure(root_dir: str, ignore_patterns: Set[str]) -> str:
    structure = []
    for root, dirs, files in os.walk(root_dir):
        root_path = Path(root)
        if is_ignored(root_path, Path(root_dir), ignore_patterns):
            continue
        rel_root = root_path.relative_to(root_dir)
        level = len(rel_root.parts)
        indent = '  ' * level
        structure.append(f"{indent}- {root_path.name}/")
        for f in sorted(files):
            file_path = root_path / f
            if file_path.suffix in VALID_EXTENSIONS and not is_ignored(file_path, Path(root_dir), ignore_patterns):
                structure.append(f"{indent}  - {f}")
    return '\n'.join(structure)

def clean_imports_and_comments(code: str) -> str:
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r'#.*', '', line)
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('import') or stripped.startswith('from'):
            if any(mod in stripped for mod in STANDARD_MODULES):
                continue
        cleaned.append(line.rstrip())
    return '\n'.join(cleaned)

def compact_python_code(code: str) -> str:
    lines = code.splitlines()
    cleaned = []
    previous_blank = False
    for line in lines:
        if line.strip() == '':
            if not previous_blank:
                cleaned.append('')
                previous_blank = True
        else:
            cleaned.append(line)
            previous_blank = False
    return '\n'.join(cleaned)

def summarize_functions_only(code: str) -> str:
    lines = code.splitlines()
    summarized = []
    inside_def = False
    called_funcs = set()

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def '):
            if inside_def and called_funcs:
                summarized[-1] += f"  # calling: {', '.join(sorted(called_funcs))}"
            summarized.append(line.rstrip() + ' pass')
            inside_def = True
            called_funcs.clear()
            continue

        if inside_def:
            func_call_match = re.match(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\(', stripped)
            if func_call_match:
                func_name = func_call_match.group(1)
                if func_name not in ('self', 'super') and not any(func_name.startswith(p.rstrip('.')) for p in STANDARD_MODULES):
                    called_funcs.add(func_name)
            current_indent = len(line) - len(line.lstrip())
            if stripped == '' or current_indent > 0:
                continue
            else:
                if called_funcs:
                    summarized[-1] += f"  # calling: {', '.join(sorted(called_funcs))}"
                inside_def = False
        else:
            summarized.append(line.rstrip())

    if inside_def and called_funcs:
        summarized[-1] += f"  # calling: {', '.join(sorted(called_funcs))}"
    return '\n'.join(summarized)

def clean_code(code: str, summary_level: str = 'high') -> str:
    if summary_level == 'none':
        return code
    code = clean_imports_and_comments(code)
    code = compact_python_code(code)
    if summary_level == 'high':
        code = summarize_functions_only(code)
    return code

def collect_source_files(root_dir: str, ignore_patterns: Set[str], summary_level: str, include_templates: bool) -> List[Tuple[str, str]]:
    sources = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            file_path = Path(root) / f
            ext = file_path.suffix
            if ext not in VALID_EXTENSIONS:
                continue
            if ext == '.html' and not include_templates:
                continue
            if ext == '.html' and 'templates' not in file_path.parts:
                continue
            if is_ignored(file_path, Path(root_dir), ignore_patterns):
                continue
            with file_path.open('r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if ext == '.py':
                    content = clean_code(content, summary_level=summary_level)
                rel_path = file_path.relative_to(root_dir)
                sources.append((str(rel_path), content))
    return sorted(sources, key=lambda item: (0 if item[0].endswith('main.py') else 1, item[0]))

def write_markdown(output_file: str, sources: List[Tuple[str, str]], structure: str = None):
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write('# Python Project Summary\n\n')
        if structure:
            out.write('## Folder Structure\n')
            out.write('```text\n')
            out.write(structure)
            out.write('\n```\n\n')
        for path, code in sources:
            out.write(f'## `{path}`\n')
            out.write('```python\n' if path.endswith('.py') else '```html\n')
            out.write(code)
            out.write('\n```\n\n')

def main():
    parser = argparse.ArgumentParser(description="Summarize a Python project into a Markdown file.")
    parser.add_argument('input_dir', help='Path to the project directory')
    parser.add_argument('-o', '--output_file', help='Output Markdown file')
    parser.add_argument('--include_structure', action='store_true', help='Include folder structure in the output')
    parser.add_argument('--include_templates', action='store_true', help='Include .html template files')
    parser.add_argument('--summary', choices=['high', 'mid', 'none'], default='none', help='Summary level: high, mid, none')
    args = parser.parse_args()

    app_name = get_app_name(args.input_dir)
    output_file = "docs/" + (args.output_file or f"{app_name}_summary_{args.summary}.md")

    ignore_patterns = load_gitignore_patterns(args.input_dir)
    structure = get_folder_structure(args.input_dir, ignore_patterns) if args.include_structure else None
    sources = collect_source_files(args.input_dir, ignore_patterns, summary_level=args.summary, include_templates=args.include_templates)
    write_markdown(output_file, sources, structure)
    print(f"✅ Project summary written to: {output_file}")

if __name__ == '__main__':
    main()
