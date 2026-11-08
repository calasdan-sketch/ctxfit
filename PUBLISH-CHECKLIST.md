# Publish checklist for ctxfit (plain words, nothing here has been run for you)

**Version to release: 0.8.0.** It is already set in the code, the changelog and the test
files. PyPI only has 0.1.1 today (put there on 2026-10-20), so people who type `pip install ctxfit`
get an old version that does not match the README. Releasing 0.8.0 fixes that. Nothing breaks for
people who use 0.1.1: every old name still works, new options all have safe defaults, and the
report files only gained fields. The number stays 0.8.0 because nothing with this version number
has ever been uploaded; PyPI will not let you upload the same number twice, so if you upload and
then change anything, the next number is 0.8.1.

## Decide first (two minutes)

1. **Is the GitHub repository public?** The package page will link to
   `https://github.com/calasdan-sketch/ctxfit` (Repository, Issues, Changelog). Open that address in
   a private browser window. If it says "404", either make the repository public first or delete
   the three lines starting `Repository`, `Issues`, `Changelog` under `[project.urls]` in
   `pyproject.toml` (keep `Homepage`). I could not check this from where I work.
2. **Is "Calas Automations Inc." the name you want on the package page?** It is the Author and
   Maintainer, with no email address (none was given, so none was made up).

## Check first (5 minutes)

1. Push the branch to GitHub and open the "Actions" tab. Wait for the `proof` workflow.
   Three jobs matter: `matrix` (Python 3.10, 3.13 and **Windows**), `package`, and `proof`.
   The Windows job is the first time this code touches Windows. If it is red, do not say
   "works on Windows" anywhere. (Python 3.10 has also never been run by me; it is in that matrix.)
2. Read `README.md` top to bottom once. The first thing a stranger sees is `ctxfit demo`.
3. Optional but cheap: `python scripts/mutation_check.py` (a few minutes). It breaks the code on
   purpose, one condition at a time, and the tests must notice. The last line should say
   "33 of 35 mutants noticed" (the two it misses are explained in `docs/ROUND5.md`).

## Build and look inside (on your computer, in a throwaway folder)

```
python -m venv /tmp/build-env
/tmp/build-env/bin/pip install build twine
/tmp/build-env/bin/python -m build
/tmp/build-env/bin/twine check dist/*
```

Both lines from `twine check` must say PASSED. Then look at the name that will be published:

```
unzip -p dist/ctxfit-0.8.0-py3-none-any.whl ctxfit-0.8.0.dist-info/METADATA | grep -E "^(Version|Author|Maintainer)"
```

It must print `Version: 0.8.0`, `Author: Calas Automations Inc.` and `Maintainer: Calas Automations Inc.`.
Then try the file you are about to upload:

```
python -m venv /tmp/clean-env
/tmp/clean-env/bin/pip install dist/ctxfit-0.8.0-py3-none-any.whl
/tmp/clean-env/bin/ctxfit --version      # must print: ctxfit 0.8.0
/tmp/clean-env/bin/ctxfit demo           # must end with three "kept:" lines
```

## Upload (this is the step that spends your name on the public registry; it is yours to do)

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
- **npm**: the README used to say there is an npm package. There is not (checked 2026-11-07, the
  name returns "not found"). The README no longer claims it.
- **GitHub secret scanning**: the test files do not hold key-shaped text, so the push should not be
  blocked. If GitHub still complains, tell me which line.
- **Claude Code hook**: a ready-made hook example is not included, because I could not confirm
  from the official documentation what a hook is allowed to return; I will not guess. Reading a
  Claude Code transcript with `ctxfit fit session.jsonl` does work.
