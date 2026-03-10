# Releasing Octavian

`octavian` is published to PyPI from Git tags via GitHub Actions using PyPI Trusted Publishing.

## One-time setup

1. Create the `octavian` project on PyPI if it does not exist yet.
2. In PyPI, configure a trusted publisher for this GitHub repository and the `.github/workflows/publish.yml` workflow.
3. Only tag releases from the branch you treat as release-ready after CI is green.

## Release steps

1. Update `version` in `pyproject.toml`.
2. Commit and merge the version bump to your release branch.
3. Create and push an annotated tag that matches the version:

```bash
git tag -a v0.1.3 -m "Release v0.1.3"
git push origin v0.1.3
```

4. GitHub Actions builds the sdist and wheel, smoke-tests the wheel install, and publishes to PyPI.

## Local verification

Run these checks before tagging:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m pip install dist/*.whl
python -c "import octavian; print(octavian.__name__)"
pytest
```

Confirm that the built artifacts contain `octavian/viz/data/cartoon_earth_map.png` and `octavian/viz/data/earth_map.jpg`.
