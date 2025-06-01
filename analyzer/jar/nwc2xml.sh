#!/bin/bash

# Bash wrapper for Nwc2MusicXML converter

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <InputFile> <OutputFile>"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR_PATH="$SCRIPT_DIR/nwc2musicxml.jar"

# Check if jar exists
if [ ! -f "$JAR_PATH" ]; then
    echo "Error: nwc2musicxml.jar not found in $SCRIPT_DIR"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

echo "Converting $INPUT_FILE to $OUTPUT_FILE..."

# Run the Java converter
java -cp "$JAR_PATH" fr.lasconic.nwc2musicxml.convert.Nwc2MusicXML "$INPUT_FILE" "$OUTPUT_FILE"

# Check result
if [ "$?" -eq 0 ]; then
    echo "Conversion completed successfully."
else
    echo "Conversion failed."
    exit 1
fi
