
import argparse
from .converter import nwc_to_simplified_abc

def main():
    parser = argparse.ArgumentParser(description="Convert NWCtxt to ABC notation")
    parser.add_argument("input_file", help="Path to NWCtxt file")
    parser.add_argument("--method", choices=["jar", "service"], default="jar", help="Conversion method")
    parser.add_argument("--jar", default="./nwc2musicxml.jar", help="Path to nwc2musicxml.jar")
    parser.add_argument("--level", choices=["raw", "medium", "simple"], default="raw", help="ABC simplicity level")
    args = parser.parse_args()
    abc = nwc_to_simplified_abc(args.input_file, method=args.method, jar_path=args.jar, simplicity_level=args.level)
    print(abc)
