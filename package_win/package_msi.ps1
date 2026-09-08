# Builds the Windows MSI installer (cx_Freeze) using the local venv_win environment.
$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "venv_win\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Could not find virtual environment python at '$python'." -ForegroundColor Red
    Write-Host "Create it with: python -m venv package_win\venv_win"
    exit 1
}

# Bring the venv up to date with requirements.txt first. cx_Freeze bundles what it
# finds installed, and a dependency missing from the venv does not fail the build --
# it drops silently out of the package and the frozen application raises ImportError
# the first time it reaches the code that needs it.
$requirements = Join-Path (Split-Path -Parent $PSScriptRoot) "requirements.txt"
& $python "-m" "pip" "install" "--quiet" "--requirement" $requirements
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not install the requirements from '$requirements'." -ForegroundColor Red
    exit $LASTEXITCODE
}

# setup.py lives here and anchors its source paths to the project root, so run the
# build from this folder -- that way build/ and dist/ land under package_win/,
# keeping the repo root clean.
Set-Location -Path $PSScriptRoot
& $python "setup.py" "bdist_msi"
exit $LASTEXITCODE
