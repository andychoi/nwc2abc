param (
    [string]$NwcSourceDir = "nwcoriginal",
    [string]$OutputRootDir = ".\converted",
    [switch]$Force,
    [string]$Steps = ""
)

$ToolPath = "C:\app\music"

# Derived folders
$NwctxtDir = Join-Path $OutputRootDir "nwctxt"
$NwctxtUtf8Dir = "$NwctxtDir-fixed"
$MusicxmlDir = Join-Path $OutputRootDir "musicxml"

$StepTitles = @{
    "1" = "🎼 Step 1: Converting .nwc → .nwctxt"
    "2" = "🛠️  Step 2: Fixing Korean mojibake in .nwctxt → $NwctxtUtf8Dir"
    "3" = "🧪 Step 3: Applying general fixes to .nwctxt"
    "4" = "🎶 Step 4: Converting .nwctxt → .musicxml"
    "5" = "🧹 Step 5: Organizing .musicxml by composer"
    "6" = "🪄 Step 6: Converting .musicxml → .abc"
}

$AllSteps = "123456"

function ShowSteps {
    Write-Host "`n📋 Available Steps:"
    foreach ($key in ($StepTitles.Keys | Sort-Object)) {
        Write-Host "$key. $($StepTitles[$key])"
    }
    Write-Host "`n💡 To run all steps:   .\convert-all.ps1 --steps all"
    Write-Host "💡 To run some steps:  .\convert-all.ps1 --steps 135"
}

function ShouldRunStep($step) {
    return $Steps.Contains("$step")
}

# Normalize --steps all
if ($Steps.ToLower() -eq "all") {
    $Steps = $AllSteps
}

# If no steps specified, show help
if (-not $Steps) {
    ShowSteps
    return
}

# Step 1
if (ShouldRunStep 1) {
    Write-Host "`n$($StepTitles['1'])" -ForegroundColor Cyan
    $step1Args = @("`"$NwcSourceDir`"", "`"$NwctxtDir`"")
    if ($Force) { $step1Args += "--force" }
    & python "$ToolPath\nwc2nwctxt.py" @step1Args    
}

# Step 2
if (ShouldRunStep 2) {
    Write-Host "`n$($StepTitles['2'])" -ForegroundColor Cyan
    $forcePy = if ($Force) { "--force" } else { "" }
    python "$ToolPath\fix-korean.py" "$NwctxtDir" "$NwctxtUtf8Dir" $forcePy
}

# Step 3
if (ShouldRunStep 3) {
    Write-Host "`n$($StepTitles['3'])" -ForegroundColor Cyan
    $step3Args = @("$NwctxtUtf8Dir")
    if ($Force) { $step3Args += "--force" }
    & python "$ToolPath\nwctxt_fix.py" @step3Args
}

# Step 4
if (ShouldRunStep 4) {
    Write-Host "`n$($StepTitles['4'])" -ForegroundColor Cyan
    $step4Args = @{
        SourceDir = $NwctxtUtf8Dir
        DestDir   = $MusicxmlDir
    }
    if ($Force) { $step4Args["Force"] = $true }
    & "$ToolPath\nwctxt2musicxml.ps1" @step4Args
}

# Step 5
if (ShouldRunStep 5) {
    Write-Host "`n$($StepTitles['5'])" -ForegroundColor Cyan
    & "$ToolPath\musicxml-organize.ps1" "$MusicxmlDir"
}

# Step 6
if (ShouldRunStep 6) {
    Write-Host "`n$($StepTitles['6'])" -ForegroundColor Cyan
    $step6Args = @("$MusicxmlDir")
    if ($Force) { $step6Args += "--force" }
    & python "$ToolPath\musicxml2abc.py" @step6Args
}

Write-Host "`n✅ Done: Steps [$Steps] completed." -ForegroundColor Green