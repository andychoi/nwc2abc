# nwc2abc/cli.py
"""
nwc2abc-cli nwc2musicxml myscore.nwc.txt -o myscore.musicxml
nwc2abc-cli nwc2abc myscore.nwc.txt -o myscore.abc
nwc2abc-cli abc2musicxml myscore.abc -o myscore.musicxml
nwc2abc-cli abc2nwc myscore.abc -o myscore.nwc.txt

"""
import argparse
from pathlib import Path
from .converter import (
    nwc_to_musicxml_jar, nwc_to_musicxml_service, nwc_to_simplified_abc,
    simplified_abc_to_musicxml, abc_to_nwc, log
)

def main():
    parser = argparse.ArgumentParser(description="nwc2abc converter toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common input/output
    def add_common_args(p):
        p.add_argument("input_file", help="Input file path")
        p.add_argument("-o", "--output", type=str, help="Output file path (optional)")

    # NWC → MusicXML
    p1 = subparsers.add_parser("nwc2musicxml", help="Convert NWCtxt to MusicXML")
    add_common_args(p1)
    p1.add_argument("--method", choices=["jar", "service"], default="jar", help="Conversion method")
    p1.add_argument("--script", default="./jar/nwc2xml.sh", help="Path to nwc2xml.sh (for jar method)")

    # NWC → ABC
    p2 = subparsers.add_parser("nwc2abc", help="Convert NWCtxt to simplified ABC")
    add_common_args(p2)
    p2.add_argument("--method", choices=["jar", "service"], default="jar")
    p2.add_argument("--script", default="./jar/nwc2xml.sh")
    p2.add_argument("--level", choices=["raw", "medium", "simple"], default="raw")

    # ABC → MusicXML
    p3 = subparsers.add_parser("abc2musicxml", help="Convert ABC to MusicXML")
    add_common_args(p3)

    # ABC → NWC
    p4 = subparsers.add_parser("abc2nwc", help="Convert ABC to NWCtxt")
    add_common_args(p4)

    args = parser.parse_args()

    if args.command == "nwc2musicxml":
        if args.method == "jar":
            output = nwc_to_musicxml_jar(args.input_file, args.script)
        else:
            output = nwc_to_musicxml_service(args.input_file)
        if args.output:
            Path(args.output).write_text(Path(output).read_text(encoding="utf-8"), encoding="utf-8")
            log(f"[INFO] MusicXML written to {args.output}")
        else:
            print(output)

    elif args.command == "nwc2abc":
        abc = nwc_to_simplified_abc(args.input_file, method=args.method, script_path=args.script, simplicity_level=args.level)
        if args.output:
            Path(args.output).write_text(abc, encoding="utf-8")
            log(f"[INFO] ABC written to {args.output}")
        else:
            print(abc)

    elif args.command == "abc2musicxml":
        abc_text = Path(args.input_file).read_text()
        output = simplified_abc_to_musicxml(abc_text, args.output)
        log(f"[INFO] MusicXML created from ABC: {output}")

    elif args.command == "abc2nwc":
        abc_text = Path(args.input_file).read_text()
        output = abc_to_nwc(abc_text, args.output)
        log(f"[INFO] NWCtxt created from ABC: {output}")

if __name__ == "__main__":
    main()
