# Model-free regression test for Windows native process management.
# Checks successful exit 0, deliberate exit 7 and timeout with redirected logs.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "galaxy-process.ps1")

$cmdExe = (Get-Command "cmd.exe" -CommandType Application -ErrorAction Stop).Source
$dir = Join-Path ([System.IO.Path]::GetTempPath()) ("galaxy-runner-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $dir -Force | Out-Null
try {
    $okLog = Join-Path $dir "ok.stdout"
    $ok = Invoke-GalaxyProcess -Executable $cmdExe -ArgumentList @("/d", "/c", "echo GALAXY_RUNNER_OK") -StdoutPath $okLog -StderrPath (Join-Path $dir "ok.stderr") -TimeoutSec 10
    if ($ok.TimedOut -or $ok.ExitCode -ne 0 -or
            -not (Select-String -Path $okLog -SimpleMatch -Pattern "GALAXY_RUNNER_OK" -Quiet)) {
        throw "Runner did not capture exit code 0 and stdout"
    }

    $bad = Invoke-GalaxyProcess -Executable $cmdExe -ArgumentList @("/d", "/c", "exit /b 7") -StdoutPath (Join-Path $dir "bad.stdout") -StderrPath (Join-Path $dir "bad.stderr") -TimeoutSec 10
    if ($bad.TimedOut -or $bad.ExitCode -ne 7) {
        throw "Runner did not capture nonzero exit code 7"
    }

    $slow = Invoke-GalaxyProcess -Executable $cmdExe -ArgumentList @("/d", "/c", "ping -n 4 127.0.0.1 >nul") -StdoutPath (Join-Path $dir "slow.stdout") -StderrPath (Join-Path $dir "slow.stderr") -TimeoutSec 1
    if (-not $slow.TimedOut) {
        throw "Runner did not enforce a one-second timeout"
    }

    Write-Host "GALAXY_PROCESS_SELFTEST PASS exit0=0 exit7=7 timeout=1"
}
finally {
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
}
