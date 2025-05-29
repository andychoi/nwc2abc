# nwctxt2musicxml.ps1
param (
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [string]$DestDir = "musicxml",

    [switch]$Force
)

# Resolve input and output paths
$NWC_TXT_DIR = Resolve-Path -Path $SourceDir -ErrorAction Stop

$MUSICXML_DIR = Resolve-Path -Path $DestDir -ErrorAction SilentlyContinue
if (-not $MUSICXML_DIR) {
    $MUSICXML_DIR = Join-Path -Path (Get-Location) -ChildPath $DestDir
    Write-Host "Creating output root folder: $MUSICXML_DIR"
    New-Item -ItemType Directory -Force -Path $MUSICXML_DIR | Out-Null
}

# Locate JAR
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$JarPath = Join-Path $ScriptDir "nwc2musicxml.jar"

if (-not (Test-Path $JarPath)) {
    Write-Error "ERROR: Converter not found at $JarPath"
    exit 1
}

# Get all nwctxt files
$nwctxtFiles = Get-ChildItem -Path $NWC_TXT_DIR -Recurse -Filter *.nwctxt

foreach ($file in $nwctxtFiles) {
    $relativePath = $file.FullName.Substring($NWC_TXT_DIR.Path.Length).TrimStart('\')
    $outputRelative = [System.IO.Path]::ChangeExtension($relativePath, ".musicxml")
    $outputFullPath = Join-Path $MUSICXML_DIR $outputRelative
    $outputDir = Split-Path $outputFullPath -Parent

    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    }

    $shouldConvert = $true
    if ((Test-Path $outputFullPath) -and (-not $Force)) {
        $srcTime = (Get-Item $file.FullName).LastWriteTimeUtc
        $dstTime = (Get-Item $outputFullPath).LastWriteTimeUtc
        if ($srcTime -le $dstTime) {
            Write-Host "Skipping (up to date): $relativePath" -ForegroundColor Yellow
            $shouldConvert = $false
        }
    }

    if ($shouldConvert) {
        # Create ASCII-safe temp input and output files
        $asciiSafeInput = Join-Path $env:TEMP ("temp_input_" + [System.IO.Path]::GetRandomFileName() + ".nwctxt")
        $asciiSafeOutput = Join-Path $env:TEMP ("temp_output_" + [System.IO.Path]::GetRandomFileName() + ".musicxml")

        # Copy-Item -Path $file.FullName -Destination $asciiSafeInput -Force
        try {
            Copy-Item -LiteralPath $file.FullName -Destination $asciiSafeInput -Force -ErrorAction Stop
        } catch {
            Write-Warning "❌ Failed to copy to temp file: $asciiSafeInput"
            continue
        }

        if (-not (Test-Path $asciiSafeInput)) {
            Write-Warning "❌ Temp input file was not created: $asciiSafeInput"
            continue
        }

        # Run Java on ASCII-safe paths
        $javaArgs = @(
            "-Dfile.encoding=UTF-8",
            "-cp", $JarPath,
            "fr.lasconic.nwc2musicxml.convert.Nwc2MusicXML",
            $asciiSafeInput,
            $asciiSafeOutput
        )

        $javaCmd = "chcp 65001 > nul && java " + ($javaArgs -join ' ')
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c $javaCmd" -Wait -NoNewWindow

        # If output succeeded, move to proper output location
        if (Test-Path $asciiSafeOutput) {
            Move-Item -Force -Path $asciiSafeOutput -Destination $outputFullPath
            Write-Host "Success: $outputFullPath" -ForegroundColor Green
        } else {
            Write-Warning "Conversion failed: $($file.FullName)"
        }

        # Clean up temp input
        if (Test-Path $asciiSafeInput) {
            Remove-Item -Force $asciiSafeInput
        }

    }


}

Write-Host "`nBatch MusicXML conversion complete." -ForegroundColor Green
