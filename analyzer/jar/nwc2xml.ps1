# PowerShell wrapper for Nwc2MusicXML converter
param (
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile
)

$jarPath = "C:\app\nwc2musicxml.jar"

if (!(Test-Path $jarPath)) {
    Write-Error "nwc2musicxml.jar not found at $jarPath"
    exit 1
}

if (!(Test-Path $InputFile)) {
    Write-Error "Input file '$InputFile' does not exist."
    exit 1
}

Write-Host "Converting $InputFile to $OutputFile..." -ForegroundColor Cyan

& java -cp $jarPath fr.lasconic.nwc2musicxml.convert.Nwc2MusicXML "$InputFile" "$OutputFile"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Conversion completed successfully." -ForegroundColor Green
} else {
    Write-Error "Conversion failed."
}
