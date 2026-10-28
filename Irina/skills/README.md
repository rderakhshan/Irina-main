# Skills

Each skill is a directory with a `SKILL.md` and optional assets or scripts:

```text
skills/
  release-notes/
    SKILL.md
  application/
    SKILL.md
    assets/cover_letter_template.tex
    scripts/finalize.py
```

```markdown
---
name: release-notes
description: Write concise release notes from git history.
---

# Release Notes

When asked for release notes:
1. Read commits since the last tag.
2. Group by feature, fix, and breaking change.
3. Keep each bullet under 120 characters.
```

Rules:

- the directory name is the skill ID (`A-Z a-z 0-9 . _ -`, max 64 characters);
- `description` is the only field shown in the system prompt;
- the full file loads only when the agent calls `use_skill("<name>")`;
- skills are lower-trust instructions and cannot grant permissions;
- files over 4,000 lines or 200,000 characters are ignored;
- symlinks and invalid names are ignored.

On startup the CLI uploads the complete skill directory, including assets and
scripts, into the OpenShell sandbox and adds the catalog to the system prompt.
Use `--skills-dir` to point elsewhere, or `--skills-dir ""` to disable loading.

Skills that need input files or produce artifacts use explicit transfers. No
host mount exists:

```powershell
uv run python main.py `
  --skills-dir .\skills `
  --input .\input-cv\resume.tex `
  --export application `
  --export-contents `
  --artifacts-dir "D:\Application" `
  --sandbox-image irina/application-tex:1 `
  --mode yolo -p "Prepare the application"
```

- `--input` uploads host files or directories into `input/`.
- `--export` names a workspace path downloaded on clean shutdown.
- `--artifacts-dir` receives those paths.
- `--sandbox-image` selects an approved image with extra tools.

The included `application` skill needs the TeX/Poppler/pypdf image in
[images/application](../images/application/README.md), and its `SKILL.md`
expects `input/resume.tex` plus optional `input/reference.pdf` and
`input/examples/`.

Skills created or edited by Irina during a run are captured as drafts under
`<state>/skill-drafts/<workspace-key>/`. Review them before promoting:

```powershell
uv run python main.py skills list
uv run python main.py skills promote <name>
uv run python main.py skills discard <name>
```

For a flag-free `uv run python main.py`, set the same values in the ignored
`.env` file (`IRINA_SANDBOX_IMAGE`, `IRINA_INPUTS`,
`IRINA_OPTIONAL_INPUTS`, `IRINA_EXPORT_PATHS`, `IRINA_ARTIFACTS_DIR`,
`IRINA_AUTO_EXPORT`, and `IRINA_MODE`). Explicit flags still override.

Promotion refuses to overwrite an existing trusted skill unless `--force` is
used. This review gate prevents one run from rewriting the instructions for
every later run.
