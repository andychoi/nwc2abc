param (
    [Parameter(Mandatory = $true)]
    [string]$RootFolder,

    [string]$OutputFolder = ""
)

# 1️⃣ Resolve and validate paths
$RootPath = Resolve-Path -Path $RootFolder -ErrorAction Stop

if (-not $OutputFolder) {
    $OutputFolder = "$($RootPath.Path)-organized"
}
$OutputPath = Resolve-Path -Path $OutputFolder -ErrorAction SilentlyContinue
if (-not $OutputPath) {
    Write-Host "📁 Creating output folder: $OutputFolder"
    New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null
}
$OutputPath = Resolve-Path $OutputFolder

Write-Host "`n📂 Organizing .musicxml files from:`n  $($RootPath.Path)" -ForegroundColor Cyan
Write-Host "📁 into:`n  $($OutputPath.Path)`n"

# 2️⃣ Gather all .musicxml files
$MusicXMLFiles = Get-ChildItem -Path $RootPath -Recurse -Filter *.musicxml
if ($MusicXMLFiles.Count -eq 0) {
    Write-Warning "⚠️  No .musicxml files found under $RootPath"
    exit 0
}

# 3️⃣ Process each file
foreach ($file in $MusicXMLFiles) {
    try {
        $escapedPath = $file.FullName.Replace("'", "''")

        $tempScript = [System.IO.Path]::GetTempFileName() + ".py"

        @"
from music21 import converter
import sys

try:
    score = converter.parse(r'''$escapedPath''')
    composer = (score.metadata.composer or '').strip() or 'unknown'
except Exception:
    composer = 'unknown'

sys.stdout.write(composer)
"@ | Set-Content -Path $tempScript -Encoding UTF8

        $composer = & python $tempScript
        Remove-Item -Force $tempScript

        # Clean and normalize folder name
        # $composer = $composer -replace '[\\/:*?"<>|]', '' -replace '\s+', '_'
        # Normalize composer name
        $composer = $composer.ToLowerInvariant()
        $composer = $composer -replace '[\._]', ' '          # Convert dots/underscores to spaces
        $composer = $composer -replace '[\\/:*?"<>|]', ''     # Remove illegal characters
        $composer = $composer -replace '\s+', ' '             # Collapse multiple spaces
        $composer = $composer.Trim()

        # Optional: title case for readability
        $composer = -join ($composer -split '\s+') | ForEach-Object { ($_ -replace '^.', { $_.Value.ToUpper() }) }

        # Fallback if empty
        if (-not $composer) { $composer = 'Unknown' }

        $targetDir = Join-Path $OutputPath $composer
        if (-not (Test-Path $targetDir)) {
            Write-Host "📁 Creating folder: $composer"
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }

        $destPath = Join-Path $targetDir $file.Name
        if ($file.FullName -ne $destPath) {
            Write-Host "📦 Moving '$($file.Name)' → '$composer\'"
            Move-Item -LiteralPath $file.FullName -Destination $destPath -Force
        }

    } catch {
        Write-Warning "⚠️  Error processing '$($file.FullName)': $_"
    }
}

Write-Host "`n✅ Organization complete." -ForegroundColor Green
