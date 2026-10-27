# Opik local setup (Windows)

The Opik source is included at `src/back/opik`. It provides the local tracing and
evaluation UI used by this project. It runs with Docker Compose.

From the repo root:

```powershell
Set-Location src/back/opik
powershell -ExecutionPolicy Bypass -File .\opik.ps1
```

## Windows PowerShell 5.1 note

The upstream `opik.ps1` is UTF-8 without a BOM. Windows PowerShell 5.1 reads
BOM-less scripts with the system code page, which can break the box-drawing and
emoji characters. You may see errors such as:

```
Array index expression is missing or not valid.
The string is missing the terminator: ".
```

Use PowerShell 7+:

```powershell
pwsh -ExecutionPolicy Bypass -File .\opik.ps1
```

Or re-save `opik.ps1` as UTF-8 with a BOM:

```powershell
$p = ".\opik.ps1"
$b = [IO.File]::ReadAllBytes($p)
[IO.File]::WriteAllBytes($p, [byte[]](0xEF, 0xBB, 0xBF) + $b)
```

Then run it with Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\opik.ps1
```

## Common commands

```powershell
powershell -ExecutionPolicy Bypass -File .\opik.ps1            # start full stack
powershell -ExecutionPolicy Bypass -File .\opik.ps1 --infra    # infrastructure only
powershell -ExecutionPolicy Bypass -File .\opik.ps1 --verify   # health check
powershell -ExecutionPolicy Bypass -File .\opik.ps1 --stop     # stop services
```

Back to the [project README](../README.md). For upstream Opik documentation,
see [src/back/opik/README.md](../src/back/opik/README.md).
