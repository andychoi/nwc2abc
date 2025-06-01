@echo off
REM Wrapper for Nwc2MusicXML converter

IF "%~1"=="" (
    echo Usage: nwc2musicxml inputfile.nwctxt outputfile.xml
    exit /b 1
)

IF "%~2"=="" (
    echo Usage: nwc2musicxml inputfile.nwctxt outputfile.xml
    exit /b 1
)

java -cp "C:\app\nwc2musicxml.jar" fr.lasconic.nwc2musicxml.convert.Nwc2MusicXML "%~1" "%~2"
