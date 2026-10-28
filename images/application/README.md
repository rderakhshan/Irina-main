# Application sandbox image

This image adds the document toolchain used by the `application` skill:

- TeX Live (`pdflatex`, `latexmk` dependencies)
- Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`)
- `pypdf` for link-preserving merges
- Python 3.12

Build it once in the same Docker environment OpenShell uses:

```powershell
wsl docker build -t irina/application-tex:1 .
```

Run Irina with it:

```powershell
uv run python main.py `
  --sandbox-image irina/application-tex:1 `
  --skills-dir .\skills `
  --input .\input-cv\resume.tex `
  --export application `
  --export-contents `
  --artifacts-dir "D:\Application" `
  --mode yolo -p "Prepare an application for this job description"
```

Verified in this checkout: `pdflatex` compiled a CV and a one-page letter,
`pypdf` merged them into a two-page `application.pdf`, and Poppler reported the
page count.

The image is trusted supply chain: treat changes to this Dockerfile like changes
to the sandbox policy. The base sandbox policy still applies; no network is
allowed at runtime.

Verify the toolchain inside a sandbox:

```bash
command -v pdflatex pdfinfo pdftotext pdftoppm
python3 -c "import pypdf; print('pypdf ok')"
```
