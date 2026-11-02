# Start a native executable with deterministic exit status and streaming logs.
#
# Start-Process -NoNewWindow with redirected streams can produce a Process
# object whose ExitCode is $null on Windows PowerShell / PowerShell 7.
# Use System.Diagnostics.Process directly to retain the process handle.
#
# Both stdout and stderr are copied asynchronously to files. This avoids
# deadlocks when one pipe fills and keeps memory bounded under -Trace.

function Invoke-GalaxyProcess {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)][string]$Executable,
        [Parameter(Mandatory=$true)][string[]]$ArgumentList,
        [Parameter(Mandatory=$true)][string]$StdoutPath,
        [Parameter(Mandatory=$true)][string]$StderrPath,
        [int]$TimeoutSec = 0
    )

    if ($TimeoutSec -lt 0 -or $TimeoutSec -gt 2147483) {
        throw "TimeoutSec must be between 0 and 2147483 seconds"
    }

    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = (Resolve-Path -LiteralPath $Executable).Path
    # Match the quoting convention previously used by Start-Process.
    $info.Arguments = [string]::Join(" ", $ArgumentList)
    $info.WorkingDirectory = (Get-Location).ProviderPath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $info
    $outFile = $null
    $errFile = $null

    try {
        $outFile = [System.IO.File]::Create($StdoutPath)
        $errFile = [System.IO.File]::Create($StderrPath)

        if (-not $proc.Start()) {
            throw "Could not start executable: $Executable"
        }

        # Begin both drains before waiting: a full stderr pipe must not block
        # the child while the parent waits for its exit.
        $outCopy = $proc.StandardOutput.BaseStream.CopyToAsync($outFile)
        $errCopy = $proc.StandardError.BaseStream.CopyToAsync($errFile)

        $timedOut = $false
        if ($TimeoutSec -gt 0) {
            if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
                $timedOut = $true
                if (-not $proc.HasExited) {
                    try { $proc.Kill() } catch {}
                }
                $proc.WaitForExit()
            }
        } else {
            $proc.WaitForExit()
        }

        # Drain remaining output and make sure logs have been flushed before
        # returning (or reporting a timeout).
        $outCopy.GetAwaiter().GetResult()
        $errCopy.GetAwaiter().GetResult()
        $outFile.Flush()
        $errFile.Flush()

        return [pscustomobject]@{
            ExitCode = [int]$proc.ExitCode
            TimedOut = $timedOut
            ProcessId = $proc.Id
        }
    }
    finally {
        if ($proc -and -not $proc.HasExited) {
            try { $proc.Kill() } catch {}
        }
        if ($outFile) { $outFile.Dispose() }
        if ($errFile) { $errFile.Dispose() }
        if ($proc)    { $proc.Dispose() }
    }
}
