$ErrorActionPreference = "Stop"
$python = "C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$app = Join-Path $PSScriptRoot "server.py"
& $python $app
