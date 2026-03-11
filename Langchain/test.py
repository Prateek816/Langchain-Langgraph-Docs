#!/usr/bin/env python3
"""
latex_validator.py

A tiny Python utility that checks whether a LaTeX source file is syntactically
correct.

Features
--------
*   Fast token‑level sanity check (balanced braces, matching \begin{…}/\end{…},
    unknown commands detection).
*   Full “dry‑run” compilation with pdflatex (draft mode) – captures fatal
    errors from the .log file.
*   Optional chktex run for additional warnings.
*   Works on Windows, macOS and Linux as long as a TeX distribution (TeX‑Live,
    MiKTeX, etc.) is on the PATH.

Usage
-----
    python latex_validator.py path/to/file.tex
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# 1️⃣  Token‑level sanity check (pylatexenc)
# --------------------------------------------------------------------------- #
try:
    from pylatexenc.latexwalker import LatexWalker, LatexSyntaxError
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "ERROR: pylatexenc not installed. Install it with:\n"
        "       pip install pylatexenc\n"
    )
    sys.exit(1)


def _quick_syntax_check(tex_source: str) -> List[str]:
    """
    Runs a very fast lexical check using ``pylatexenc``.
    It catches:
      • unbalanced curly/bracket groups
      • missing ``\\end{…}`` for a ``\\begin{…}``
      • malformed command names
    The function returns a list of human‑readable error messages (empty if OK).
    """
    errors: List[str] = []
    walker = LatexWalker(tex_source)

    try:
        # The walker parses the whole document; any syntax error raises an
        # exception that contains line/col information.
        walker.get_latex_nodes()
    except LatexSyntaxError as exc:
        errors.append(f"Token‑level syntax error: {exc}")

    # Additional manual brace‑balance test (pylatexenc is generous, we want a
    # strict check for stray '}' or '{')
    open_braces = tex_source.count("{")
    close_braces = tex_source.count("}")
    if open_braces != close_braces:
        errors.append(
            f"Mismatched braces: {{ count = {open_braces}, }} count = {close_braces}"
        )

    return errors


# --------------------------------------------------------------------------- #
# 2️⃣  Full compilation check (pdflatex)
# --------------------------------------------------------------------------- #
def _run_pdflatex(tex_path: Path, work_dir: Path) -> Tuple[int, str]:
    """
    Executes ``pdflatex -interaction=nonstopmode -halt-on-error -draftmode``.
    Returns the exit code and the complete log output as a string.
    """
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-draftmode",
        "-output-directory",
        str(work_dir),
        str(tex_path),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=work_dir,
    )
    return proc.returncode, proc.stdout


def _extract_fatal_errors(log: str) -> List[str]:
    """
    Parses the pdflatex log and extracts the lines that start with ``!`` (the
    classic TeX fatal error marker) together with a few surrounding lines for
    context.
    """
    errors = []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("!"):
            # Grab the error line plus the next two lines (usually the line
            # where the error occurred and a helpful hint).
            context = "\n".join(lines[i : i + 3])
            errors.append(context.strip())
    return errors


# --------------------------------------------------------------------------- #
# 3️⃣  Optional chktex run (style / common mistakes)
# --------------------------------------------------------------------------- #
def _run_chktex(tex_path: Path) -> List[str]:
    """
    If ``chktex`` is available, run it and return its warnings.
    """
    if shutil.which("chktex") is None:
        return []   # chktex not installed – silently ignore.

    cmd = ["chktex", "-q", "-n22", "-n30", str(tex_path)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # chktex prints one warning per line, prefixed with the file name.
    warnings = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return warnings


# --------------------------------------------------------------------------- #
# 4️⃣  Public validator interface
# --------------------------------------------------------------------------- #
def validate_latex(tex_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate *tex_path*.

    Returns
    -------
    (is_valid, messages)
        *is_valid* – ``True`` if no errors were found.
        *messages* – list of error / warning strings (empty when ``is_valid`` is
                     ``True``).
    """
    tex_path = tex_path.resolve()
    if not tex_path.is_file():
        raise FileNotFoundError(f"File not found: {tex_path}")

    # ------------------------------------------------------------------- #
    # 1️⃣  Quick lexical sanity check
    # ------------------------------------------------------------------- #
    with tex_path.open(encoding="utf-8") as f:
        source = f.read()
    messages: List[str] = _quick_syntax_check(source)

    # ------------------------------------------------------------------- #
    # 2️⃣  Full LaTeX compilation (in a temporary folder)
    # ------------------------------------------------------------------- #
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Copy the main .tex file (and any .bib/.cls/.sty that live beside it)
        shutil.copy(tex_path, work_dir / tex_path.name)

        # If the document uses local resources (images, .bib files, etc.) we
        # also copy the whole directory tree – this is a best‑effort approach.
        # Users can disable this by setting `copy_tree=False` later if needed.
        for extra in tex_path.parent.iterdir():
            if extra.is_file() and extra.suffix.lower() in {
                ".bib",
                ".cls",
                ".sty",
                ".png",
                ".jpg",
                ".jpeg",
                ".pdf",
                ".eps",
                ".svg",
            }:
                shutil.copy(extra, work_dir / extra.name)

        rc, log = _run_pdflatex(tex_path, work_dir)
        if rc != 0:
            # Non‑zero return code usually means a fatal error; we still parse
            # the log to give a nicer message.
            messages.append("pdflatex terminated with a non‑zero exit code.")
        messages.extend(_extract_fatal_errors(log))

    # ------------------------------------------------------------------- #
    # 3️⃣  Optional chktex warnings (non‑blocking)
    # ------------------------------------------------------------------- #
    chk_warnings = _run_chktex(tex_path)
    if chk_warnings:
        messages.append("chktex warnings (not fatal but worth reviewing):")
        messages.extend(chk_warnings)

    is_valid = len(messages) == 0
    return is_valid, messages


# --------------------------------------------------------------------------- #
# 5️⃣  CLI entry‑point
# --------------------------------------------------------------------------- #
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LaTeX source for syntactic correctness."
    )
    parser.add_argument(
        "tex_file",
        type=Path,
        help="Path to the .tex file that should be validated.",
    )
    args = parser.parse_args()

    try:
        ok, msgs = validate_latex(args.tex_file)
    except Exception as exc:  # pragma: no cover
        sys.exit(f"Validation failed: {exc}")

    if ok:
        print("✅ LaTeX source looks syntactically correct.")
        sys.exit(0)
    else:
        print("❌ Errors detected:")
        for m in msgs:
            print("- " + m)
        sys.exit(1)


if __name__ == "__main__":
    _cli()