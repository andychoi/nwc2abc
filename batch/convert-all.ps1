param (
    [string]$NwcSourceDir = "nwcoriginal",
    [string]$OutputRootDir = ".\converted",
    [switch]$Force,
    [string]$Steps = "12345"  # Default: all steps
)

# Script location
$ToolPath = "C:\app\music"

# Derived folders
$NwctxtDir = Join-Path $OutputRootDir "nwctxt"
$NwctxtUtf8Dir = "$NwctxtDir-utf8"
$MusicxmlDir = Join-Path $OutputRootDir "musicxml"

function ShouldRunStep($step) {
    return $Steps.Contains("$step")
}

# Step 1
if (ShouldRunStep 1) {
    Write-Host "`n🎼 Step 1: Converting .nwc → .nwctxt..." -ForegroundColor Cyan
    # $step1Args = @{
    #     SourceDir = $NwcSourceDir
    #     DestDir   = $NwctxtDir
    # }
    # if ($Force) { $step1Args["Force"] = $true }
    # & "$ToolPath\nwc2nwctxt.ps1" @step1Args
    $step1Args = @()
    $step1Args += "`"$NwcSourceDir`""
    $step1Args += "`"$NwctxtDir`""
    if ($Force) { $step1Args += "--force" }
    & python "$ToolPath\nwc2nwctxt.py" @step1Args    
}

# Step 2
if (ShouldRunStep 2) {
    Write-Host "`n🛠️  Step 2: Fixing Korean mojibake in .nwctxt → $NwctxtUtf8Dir..." -ForegroundColor Cyan
    $forcePy = if ($Force) { "--force" } else { "" }
    python "$ToolPath\fix-korean.py" "$NwctxtDir" "$NwctxtUtf8Dir" $forcePy
}

# Step 3
if (ShouldRunStep 3) {
    Write-Host "`n🎶 Step 3: Converting .nwctxt → .musicxml..." -ForegroundColor Cyan
    $step3Args = @{
        SourceDir = $NwctxtUtf8Dir
        DestDir   = $MusicxmlDir
    }
    if ($Force) { $step3Args["Force"] = $true }
    & "$ToolPath\nwctxt2musicxml.ps1" @step3Args
}

# Step 4
if (ShouldRunStep 4) {
    Write-Host "`n🧹 Step 4: Organizing .musicxml by composer..." -ForegroundColor Cyan
    & "$ToolPath\musicxml-organize.ps1" "$MusicxmlDir"
}

# Step 5
if (ShouldRunStep 5) {
    Write-Host "`n🪄 Step 5: Converting .musicxml → .abc..." -ForegroundColor Cyan
    $step5Args = @("$MusicxmlDir")
    if ($Force) { $step5Args += "--force" }
    & python "$ToolPath\musicxml2abc.py" @step5Args
}

Write-Host "`n✅ Done: Steps [$Steps] completed." -ForegroundColor Green
