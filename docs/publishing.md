# Publishing With GitHub Pages

This repository includes a GitHub Actions workflow for publishing the MkDocs
site to GitHub Pages from `dev`.

## One-Time Repository Setting

In GitHub:

1. Open repository settings.
2. Go to **Pages**.
3. Set **Source** to **GitHub Actions**.

After that setting is enabled, pushes to `dev` publish the site automatically.

## Manual Publish

The docs workflow also supports manual runs:

1. Open the **Actions** tab.
2. Select **Docs**.
3. Choose **Run workflow**.

## Local Verification

Before merging docs changes:

```bash
pip install -e ".[dev]"
python -m mkdocs build
```

For a local browser preview:

```bash
python -m mkdocs serve
```

## Published URL

The configured site URL is:

```text
https://ghevia.github.io/octavian/
```
