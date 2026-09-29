# Contributing

RapidCADPy welcomes focused fixes, backend integrations, tests, examples, and
documentation. The project is alpha software, so a small change with a clear
contract and evidence is more useful than a broad compatibility claim.

## Set up a development checkout

For a standalone clone:

```bash
git clone https://github.com/rapidcad/rapidcadpy.git
cd rapidcadpy
uv sync
uv run pytest
```

Install only the extras needed for your change:

```bash
uv sync --extra ocp
uv sync --extra drawing
uv sync --extra fea
```

FreeCAD is discovered from an existing installation. Autodesk Inventor tests
require Windows, Inventor, and the `inventor` extra.

RapidCADPy is also used as `vendor/rapidcadpy` in the parent project. Clone that
project with `--recurse-submodules`, or initialize it with:

```bash
git submodule update --init --recursive
```

Commit library changes inside the RapidCADPy submodule first. Then make a
separate parent-repository commit that advances only the submodule pointer.

## Repository map

- `rapidcadpy/` contains backend-neutral contracts and runtime orchestration.
- `rapidcadpy/integrations/<backend>/` contains product-specific adapters.
- `rapidcadpy/fea/` contains meshing and solver support.
- `tests/` contains contract and native integration tests.
- `examples/` contains runnable public-API examples.
- `docs/` contains design and contributor documentation.

## Architecture rules

Read [Architecture](./architecture.md) before changing a public contract or
backend boundary. Contributions must preserve these invariants:

- Keep native handles inside runtime state; never return them from `to_dict()`
  or another public serialization path.
- Use backend-neutral public references and modeling requests.
- Put product-specific logic under the corresponding integration package.
- Require an expected revision for live-document edits, then recompute,
  validate, and rehydrate before reporting success.
- Preserve native feature history when the backend supports it. Do not silently
  fall back to a direct B-Rep edit.
- Keep visible desktop-CAD connections separate from headless or isolated
  artifact execution.
- Import optional CAD dependencies lazily so the core package remains usable
  without FreeCAD, OCP, Inventor, or FEA libraries.

## Tests

The default command runs the dependency-light core suite:

```bash
uv run pytest
```

Optional tests are divided into explicit groups:

```bash
uv run pytest --test-group ocp
uv run pytest --test-group freecad
uv run pytest --test-group inventor
uv run pytest --test-group fea
```

Repeat `--test-group` to combine groups. Group selection and the documented
legacy quarantine live in [`conftest.py`](../conftest.py).

Add focused regression tests beside the affected subsystem. Tests must:

- use public contracts where the behavior is public;
- use `tmp_path`, mocks, or a fake backend instead of personal paths;
- skip unavailable native runtimes before importing their adapters;
- mark native integration requirements clearly;
- assert failure behavior for stale revisions, unsupported operations, and
  rollback when those paths are affected.

Run native tests in the application environment they require. A skipped native
test is useful only when its skip condition is specific and the same behavior
is covered by a contract test where possible.

## Quality and packaging checks

Before opening a pull request, run the checks relevant to your change:

```bash
uv run pytest
uv run ruff check conftest.py  # replace with the Python files you changed
git diff --check
uv build
```

If packaging changed, also import the wheel outside the source tree:

```bash
uv run --isolated --no-project \
  --with ./dist/rapidcadpy-0.1.0-py3-none-any.whl \
  python -c "import rapidcadpy; print(rapidcadpy.__version__)"
```

Do not commit `.venv`, caches, generated CAD files, screenshots produced by
tests, solver output, or local application settings. A documentation asset is
appropriate when it is intentional, compressed, and referenced by the docs.

## Code style

- Use four spaces and full type annotations for new public Python APIs.
- Follow Ruff naming and formatting conventions.
- Keep import-time work small and side-effect free.
- Return actionable support and error details rather than swallowing native
  exceptions or claiming a degraded operation succeeded.
- Keep examples short, runnable from a fresh clone, and based on public APIs.

## Pull requests

Use a short imperative commit summary. Keep commits narrow enough that a
reviewer can connect each change to its tests.

A pull request should explain:

- the concrete behavior before and after the change;
- the backend and lifecycle affected;
- verification commands and native environments used;
- configuration, compatibility, or migration effects;
- screenshots or representative artifacts when output changes visually.

Before requesting review, confirm that the default suite collects cleanly and
that optional dependency failures become explicit skips rather than collection
errors.
