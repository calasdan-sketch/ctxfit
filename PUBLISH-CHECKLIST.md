# Publish checklist for ctxfit (plain words, nothing here has been run)

**Version to release: 0.8.0.** It is already set in the code, the changelog and the test
files. PyPI only has 0.1.1 today, so people who type `pip install ctxfit` get an old version
that does not match the README. Releasing 0.8.0 fixes that. Nothing breaks for people who
use 0.1.1: every old name still works, new options all have safe defaults.

## Check first (5 minutes)

1. Push the branch to GitHub and open the "Actions" tab. Wait for the `proof` workflow.
   Three jobs matter: `matrix` (Python 3.10, 3.13 and **Windows**), `package`, and `proof`.
   The Windows job is the first time this code touches Windows. If it is red, do not say
   "works on Windows" anywhere.
2. Open the repo page and confirm the address is really
   `https://github.com/calasdan-sketch/ctxfit` (it is in the package links and README).
3. Read `README.md` top to bottom once. The first thing a stranger sees is `ctxfit demo`.

## Build and look inside (on your computer, in a throwaway folder)

```
python -m venv /tmp/build-env
/tmp/build-env/bin/pip install build twine
/tmp/build-env/bin/python -m build
/tmp/build-env/bin/twine check dist/*
```

Both lines from `twine check` must say PASSED. Then try the file you are about to upload:

```
python -m venv /tmp/clean-env
/tmp/clean-env/bin/pip install dist/ctxfit-0.8.0-py3-none-any.whl
/tmp/clean-env/bin/ctxfit --version      # must print: ctxfit 0.8.0
/tmp/clean-env/bin/ctxfit demo           # must end with three "kept:" lines
```

## Upload (this is the step that spends your name on the public registry)

Practice on TestPyPI first if you want a dry run:

```
/tmp/build-env/bin/twine upload --repository testpypi dist/*
```

Real upload (needs your PyPI token; PyPI will not let you re-use a version number):

```
/tmp/build-env/bin/twine upload dist/*
git tag v0.8.0
git push origin v0.8.0
```

Then, in a new empty folder: `pip install ctxfit` and `ctxfit --version` should say 0.8.0.

## Things to decide

- **Windows**: say it is supported only after the Windows job above is green.
- **npm**: the README used to say there is an npm package. There is not (checked
  2026-11-07, the name returns "not found"). The README no longer claims it.
- **GitHub secret scanning**: the test files no longer hold key-shaped text, so the push
  should not be blocked. If GitHub still complains, tell me which line.
