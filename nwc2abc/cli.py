# nwc2abc/cli.py
"""
# convert NWC → MusicXML only (print file path)
nwc2abc-cli myscore.nwc.txt --only-musicxml

# convert NWC → MusicXML only (save to file)
nwc2abc-cli myscore.nwc.txt --only-musicxml --output myscore.musicxml

# normal NWC → ABC workflow
nwc2abc-cli myscore.nwc.txt --output myscore.abc

# simplified ABC to MusicXML
nwc2abc-cli myscore.abc --from-abc --output myscore.musicxml

"""
import argparse
from pathlib import Path
from .converter import nwc_to_simplified_abc, nwc_to_musicxml_jar, nwc_to_musicxml_service, simplified_abc_to_musicxml, log

def main():
    parser = argparse.ArgumentParser(description="Convert NWCtxt to MusicXML and/or ABC notation")
    parser.add_argument("input_file", help="Path to NWCtxt file")
    parser.add_argument("--output", "-o", type=str, help="Output file (ABC or MusicXML depending on mode; else print to stdout)")
    parser.add_argument("--method", choices=["jar", "service"], default="jar", help="Conversion method (jar=local, service=online)")
    parser.add_argument("--script", default="./jar/nwc2xml.sh", help="Path to nwc2xml.sh script (used for jar method)")
    parser.add_argument("--level", choices=["raw", "medium", "simple"], default="raw", help="ABC simplicity level (ignored if only-musicxml)")
    parser.add_argument("--only-musicxml", action="store_true", help="Only output MusicXML file (no ABC conversion)")
    parser.add_argument("--from-abc", action="store_true", help="Convert simplified ABC to MusicXML (ABC input required)")

    args = parser.parse_args()

    if args.from_abc:
        abc_text = Path(args.input_file).read_text()
        output = simplified_abc_to_musicxml(abc_text, args.output)
        log(f"[INFO] MusicXML created from ABC: {output}")
    
    elif args.only_musicxml:
        # output just MusicXML
        if args.method == "jar":
            musicxml_path = nwc_to_musicxml_jar(args.input_file, script_path=args.script)
        else:
            musicxml_path = nwc_to_musicxml_service(args.input_file)

        if args.output:
            Path(args.output).write_text(Path(musicxml_path).read_text(encoding="utf-8"), encoding="utf-8")
            log(f"[INFO] MusicXML written to file: {args.output}")
        else:
            print(musicxml_path)

    else:
        # full NWC → ABC workflow
        abc = nwc_to_simplified_abc(
            args.input_file,
            method=args.method,
            script_path=args.script,
            simplicity_level=args.level
        )

        if args.output:
            Path(args.output).write_text(abc, encoding="utf-8")
            log(f"[INFO] ABC written to file: {args.output}")
        else:
            print(abc)


if __name__ == "__main__":
    main()
