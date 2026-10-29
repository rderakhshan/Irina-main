# Application sandbox image

This image provides the toolchain used by Irina's document, data, and web tools:

- TeX Live (`pdflatex`, `latexmk` dependencies) and common packages
- Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`)
- Python 3.12
- `pypdf`, `reportlab`
- `python-docx`, `openpyxl`, `python-pptx`
- `pandas`, `pyarrow`
- `nbclient`, `nbformat`, `ipykernel`
- `requests`, `SQLAlchemy`

Build it once in the same Docker environment OpenShell uses:

```powershell
wsl docker build -t irina/irina-tools:1 .\images\application
```

Run Irina with it:

```powershell
uv run python main.py `
  --sandbox-image irina/irina-tools:1 `
  --skills-dir .\skills `
  --input .\input-cv\resume.tex `
  --export application `
  --export-contents `
  --artifacts-dir "D:\Application" `
  --mode yolo -p "Prepare an application for this job description"
```

For web tools, add the opt-in network policy:

```powershell
uv run python main.py `
  --sandbox-image irina/irina-tools:1 `
  --sandbox-policy src/back/harness/tools_and_guardrails/openshell_policy_network.yaml `
  -p "Fetch https://example.com and summarize it"
```

The default policy remains strict: no network destinations are allowed. The
network policy lists each host explicitly; unlisted destinations fail with a
clear error and are never retried outside OpenShell.

The image is trusted supply chain: treat changes to this Dockerfile like changes
to the sandbox policy. Verify the toolchain inside a sandbox:

```bash
command -v pdflatex pdfinfo pdftotext pdftoppm
python3 -c "import pypdf, reportlab, docx, openpyxl, pptx, pandas, pyarrow, nbclient; print('tools ok')"
```
