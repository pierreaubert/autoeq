param(
    [Parameter(Mandatory = $true)][string]$Benchmark,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$InputFile,
    [Parameter(Mandatory = $true)][string]$OutputFile
)

$ErrorActionPreference = "Stop"
$registryPath = "HKLM:\SOFTWARE\EqualizerAPO"
$configDirectory = Split-Path -Parent (Resolve-Path $Config)
$previous = $null
$hadPrevious = $false
if (Test-Path $registryPath) {
    try {
        $previous = (Get-ItemProperty -Path $registryPath -Name ConfigPath).ConfigPath
        $hadPrevious = $true
    } catch {}
} else {
    New-Item -Path $registryPath -Force | Out-Null
}

try {
    Set-ItemProperty -Path $registryPath -Name ConfigPath -Value $configDirectory
    & $Benchmark --nopause --verbose --input $InputFile --output $OutputFile
    if ($LASTEXITCODE -ne 0) {
        throw "Equalizer APO Benchmark exited with $LASTEXITCODE"
    }
    if (-not (Test-Path $OutputFile)) {
        throw "Equalizer APO Benchmark did not create $OutputFile"
    }
} finally {
    if ($hadPrevious) {
        Set-ItemProperty -Path $registryPath -Name ConfigPath -Value $previous
    } else {
        Remove-ItemProperty -Path $registryPath -Name ConfigPath -ErrorAction SilentlyContinue
    }
}
