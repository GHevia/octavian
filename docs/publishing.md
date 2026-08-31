# Publishing With GitHub Pages

This repository has one GitHub Actions workflow, `.github/workflows/docs.yml`,
for publishing the MkDocs site to GitHub Pages from `dev`. Do not add another
Pages deployment workflow: a second deployment can overwrite the MkDocs site
with a different artifact even when both workflow runs report success.

The site uses the `readthedocs` MkDocs theme. Validate theme changes locally
with `python -m mkdocs build --strict`; this catches broken navigation,
mkdocstrings failures, and missing pages.

## One-Time Repository Setting

In GitHub:

1. Open repository settings.
2. Go to **Pages**.
3. Set **Source** to **GitHub Actions**.

After that setting is enabled, pushes to `dev` publish the site automatically.
The workflow verifies the public home and getting-started URLs after every
deployment, including retries for the short Pages propagation delay.

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

## Troubleshooting A 404

GitHub Pages can return a "site not found" page even when the docs build
locally. Check these causes:

- The docs PR has not been merged to `dev`, so the Pages workflow has not run
  from the publishing branch yet.
- Repository settings still use a branch source instead of **GitHub Actions**.
- The `Docs` workflow failed, is still queued, or failed its public URL check.
- Another workflow contains `actions/deploy-pages` and replaced the MkDocs
  artifact. There must be exactly one Pages deployment workflow.
- The repository is private or the plan/settings do not allow public Pages for
  the repo.
- The Pages deployment was disabled or replaced when the source changed from a
  previous branch-based Pages setup to GitHub Actions.
- DNS and Pages caches can lag for a few minutes after the first successful
  deployment.

Check **Actions > Docs** first. A successful run now proves both the MkDocs
build and the public URLs. If no deployment exists, enable **Settings > Pages >
Source: GitHub Actions** and run the workflow manually.
