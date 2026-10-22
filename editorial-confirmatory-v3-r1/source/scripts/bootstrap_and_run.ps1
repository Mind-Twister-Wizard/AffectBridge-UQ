$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$engine = Join-Path $root "ENGINE"
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$sha = "6c043d12e40afce2790e8f11ad67a52153ad20ac"
$logs = Join-Path $root "RUN_LOGS"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Start-Transcript -Append -Path (Join-Path $logs "latest_run.log") | Out-Null
function Check($label) { if ($LASTEXITCODE -ne 0) { throw "$label failed (exit $LASTEXITCODE)." } }
try {
  if (-not (Test-Path $py)) {
    $base = $null
    foreach($vers in @("-3.11","-3.12","-3.10")) {
      try {
        $found = & py $vers -c "import sys;print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and (Test-Path "$found")) { $base = "$found"; break }
      } catch {}
    }
    if (-not $base) { throw "Install Python 3.10, 3.11 or 3.12, then retry. The source patch is public but this launcher does not silently install an unverified interpreter." }
    & $base -m venv $venv; Check "venv creation"
  }
  if (-not (Test-Path (Join-Path $engine "run_all.py"))) {
    if (Test-Path $engine) { throw "ENGINE already exists but is incomplete; preserving it for manual inspection." }
    $archive = Join-Path $root "PINNED_UPSTREAM.zip"
    Invoke-WebRequest -Uri "https://codeload.github.com/Mind-Twister-Wizard/AffectBridge-UQ/zip/$sha" -OutFile $archive
    $unpack = Join-Path $root "_upstream_extract"
    if (Test-Path $unpack) { Remove-Item $unpack -Recurse -Force }
    Expand-Archive -Path $archive -DestinationPath $unpack
    $source = Get-ChildItem $unpack -Directory | Where-Object { Test-Path (Join-Path $_.FullName "run_all.py") } | Select-Object -First 1
    if (-not $source) { throw "Pinned source missing run_all.py" }
    Move-Item $source.FullName $engine
    Remove-Item $unpack -Recurse -Force
  }
  & $py -m pip install --disable-pip-version-check "pip<26"; Check "pip"
  & $py -m pip install --disable-pip-version-check "torch==2.6.0" --index-url "https://download.pytorch.org/whl/cu124"; Check "CUDA PyTorch"
  & $py (Join-Path $root "patches\apply_editorial_patch.py") --engine $engine --patches (Join-Path $root "patches"); Check "source patch"
  & $py -m pip install --disable-pip-version-check -r (Join-Path $engine "requirements.txt"); Check "dependencies"
  & $py -m pip install --disable-pip-version-check "transformers==4.48.3"; Check "Transformers"
  & $py -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))"; Check "CUDA visibility"
  $env:PYTHONUTF8 = "1"
  $env:TOKENIZERS_PARALLELISM = "false"
  & $py (Join-Path $root "scripts\prepare_config.py") --engine $engine; Check "config"
  Push-Location $engine
  try {
    & $py run_all.py --config editorial_config.yaml --download --train --report --device cuda --ablation-scope representative --seeds 42 52 62
    Check "full experiment"
  } finally { Pop-Location }
  & $py (Join-Path $root "scripts\check_and_package.py") --engine $engine
  Check "final validation"
  Write-Host "Verified results in ENGINE\outputs" -ForegroundColor Green
} catch {
  Write-Host $_.Exception.Message -ForegroundColor Red
  exit 1
} finally {
  try { Stop-Transcript | Out-Null } catch {}
}
