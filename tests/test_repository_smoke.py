from __future__ import annotations

import compileall
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_contract_is_present() -> None:
    required_paths = [
        "README.md",
        "requirements.txt",
        "requirements-test.txt",
        ".github/workflows/ci.yml",
        "docs/appel_offre.md",
        "docs/architecture.md",
        "docs/ethique_et_limites.md",
        "docs/evaluation_protocol.md",
        "app/app.py",
        "data/maira2_interface.csv",
        "prompts/json_schema.md",
        "prompts/medgemma_report_writer_prompt.txt",
        "prompts/medgemma_grounded_description_prompt.txt",
        "prompts/gptoss_maira2_classifier_prompt.txt",
    ]
    missing = [path for path in required_paths if not (ROOT / path).exists()]
    assert missing == []


def test_notebooks_are_present_and_valid_json() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) >= 5

    for notebook in notebooks:
        with notebook.open("r", encoding="utf-8") as file:
            data = json.load(file)
        assert "cells" in data and len(data["cells"]) > 0


def test_prompts_are_non_empty() -> None:
    for prompt_file in (ROOT / "prompts").glob("*.txt"):
        assert prompt_file.read_text(encoding="utf-8").strip() != ""


def test_non_clinical_warning_is_present_in_app_and_readme() -> None:
    warning = "Prototype pedagogique"
    app_source = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert warning in app_source
    assert "pédagogique" in readme.lower() or "pedagogique" in readme.lower()
    assert "diagnostic" in readme.lower()


def test_python_source_tree_compiles() -> None:
    for folder in ("app", "tests"):
        assert compileall.compile_dir(ROOT / folder, quiet=1)
