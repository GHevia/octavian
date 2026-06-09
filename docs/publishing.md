# Publishing With GitHub Pages

This repository includes a GitHub Actions workflow for publishing the MkDocs
site to GitHub Pages from `dev`.

The site uses the `readthedocs` MkDocs theme. Validate theme changes locally
with `python -m mkdocs build --strict`; this catches broken navigation,
mkdocstrings failures, and missing pages.

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

## Why The URL Might 404

GitHub Pages can return a "site not found" page even when the docs build
locally. Common causes:

- The docs PR has not been merged to `dev`, so the Pages workflow has not run
  from the publishing branch yet.
- Repository settings still use a branch source instead of **GitHub Actions**.
- The `Docs` workflow failed or is still queued.
- The repository is private or the plan/settings do not allow public Pages for
  the repo.
- The Pages deployment was disabled or replaced when the source changed from a
  previous branch-based Pages setup to GitHub Actions.
- DNS and Pages caches can lag for a few minutes after the first successful
  deployment.

Check **Actions > Docs** first. If the latest run succeeded, open the deployment
URL shown in that run. If no deployment exists, enable **Settings > Pages >
Source: GitHub Actions** and run the workflow manually.
