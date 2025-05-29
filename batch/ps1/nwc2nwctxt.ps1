# nwc2nwctxt.ps1
# replaced with python for better flexibility
param (
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [string]$DestDir = "nwctxt",

    [switch]$Force
)

# Get short path using Windows API
function Get-ShortPath($longPath) {
    $short = New-Object -ComObject Scripting.FileSystemObject
    return $short.GetFile($longPath).ShortPath
}

# Resolve full paths
$NWC_SOURCE_DIR = Resolve-Path -Path $SourceDir -ErrorAction Stop
$NWCTXT_DEST_DIR = Resolve-Path -Path $DestDir -ErrorAction SilentlyContinue

if (-not $NWCTXT_DEST_DIR) {
    $NWCTXT_DEST_DIR = Join-Path -Path (Get-Location) -ChildPath $DestDir
    Write-Host "Creating destination folder: $NWCTXT_DEST_DIR"
    New-Item -ItemType Directory -Force -Path $NWCTXT_DEST_DIR | Out-Null
}

# Path to Noteworthy CLI tool
$NWC2_PATH = "C:\Program Files (x86)\Noteworthy Software\NoteWorthy Composer 2\nwc2.exe"

if (-not (Test-Path $NWC2_PATH)) {
    Write-Error "ERROR: nwc2.exe not found at $NWC2_PATH"
    exit 1
}

# Find all .nwc files recursively
$nwcFiles = Get-ChildItem -Path $NWC_SOURCE_DIR -Recurse -Filter *.nwc

foreach ($file in $nwcFiles) {
    $relativePath = $file.FullName.Substring($NWC_SOURCE_DIR.Path.Length).TrimStart('\')
    $outputRelative = [System.IO.Path]::ChangeExtension($relativePath, ".nwctxt")
    $outputFullPath = Join-Path $NWCTXT_DEST_DIR $outputRelative
    $outputDir = Split-Path $outputFullPath -Parent

    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }

    $shouldConvert = $true
    if ((Test-Path -LiteralPath $outputFullPath) -and (-not $Force)) {
        $srcTime = (Get-Item -LiteralPath $file.FullName).LastWriteTimeUtc
        $dstTime = (Get-Item -LiteralPath $outputFullPath).LastWriteTimeUtc
        if ($srcTime -le $dstTime) {
            Write-Host "Skipping (up to date): $relativePath" -ForegroundColor Yellow
            $shouldConvert = $false
        }
    }

    if ($shouldConvert) {
        Write-Host "Converting: $($file.FullName) → $outputFullPath" -ForegroundColor Cyan

        try {
            $shortInput = Get-ShortPath $file.FullName
            $shortOutput = Get-ShortPath (Resolve-Path $outputDir).Path + "\" + [System.IO.Path]::GetFileName($outputFullPath)
            & "$NWC2_PATH" -convert "$shortInput" "$shortOutput"

            if (-not (Test-Path $outputFullPath)) {
                Write-Warning "WARNING: Conversion failed: $($file.FullName)"
            } else {
                Write-Host "Success: $outputFullPath" -ForegroundColor Green
            }
        } catch {
            Write-Warning "ERROR: Failed to convert $($file.FullName): $_"
        }
    }
}

Write-Host "`n✅ Batch NWC → NWCTXT conversion complete." -ForegroundColor Green
