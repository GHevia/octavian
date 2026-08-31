"""Regression tests for the hosted documentation configuration."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_only_docs_workflow_deploys_github_pages() -> None:
    """A second Pages artifact can silently overwrite the valid MkDocs site."""
    deployers = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        if "actions/deploy-pages@" in workflow.read_text(encoding="utf-8"):
            deployers.append(workflow.name)

    assert deployers == ["docs.yml"]


def test_docs_workflow_deploys_site_and_checks_public_urls() -> None:
    """The deploy job should publish MkDocs output and verify public routing."""
    workflow = (WORKFLOWS / "docs.yml").read_text(encoding="utf-8")

    assert "actions/deploy-pages@v5" in workflow
    assert "path: site" in workflow
    assert "steps.deployment.outputs.page_url" in workflow
    assert "tutorials/getting-started/" in workflow


def test_readme_documentation_links_work_outside_github() -> None:
    """PyPI renders README links without GitHub's repository-relative context."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    relative_markdown_links = re.findall(r"\]\((?!https?://|#|mailto:)([^)]+)\)", readme)

    assert relative_markdown_links == []
