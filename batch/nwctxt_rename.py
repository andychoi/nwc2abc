#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List

# Role mappings with lowercase support
ROLE_KEYWORDS = {
    "s": "Soprano", "sop": "Soprano", "soprano": "Soprano", "소프라노": "Soprano",
    "a": "Alto", "alt": "Alto", "alto": "Alto", "알토": "Alto",
    "t": "Tenor", "ten": "Tenor", "tenor": "Tenor", "테너": "Tenor",
    "b": "Bass", "bas": "Bass", "bass": "Bass", "베이스": "Bass",
    "sa": "SA", "tb": "TB",
    "rh": "Piano-RH", "right": "Piano-RH", "right hand": "Piano-RH",
    "lh": "Piano-LH", "left": "Piano-LH", "left hand": "Piano-LH",
    "피아노r": "Piano-RH", "피아노l": "Piano-LH", "피아노": "Piano",
    "violin": "Violin", "cello": "Cello", "flute": "Flute"
}

VCF = {"Violin", "Cello", "Flute"}


def normalize(text: str) -> str:
    return text.strip().lower()


def classify_by_keyword(name: str, clef: str = "") -> str:
    key = normalize(name)
    for k, role in ROLE_KEYWORDS.items():
        if key == k or k in key:
            return role
    # Brace group handled separately
    return "Piano-RH" if clef == "Treble" else "Piano-LH" if clef == "Bass" else ""


def parse_staff_blocks(lines: List[str]) -> List[List[str]]:
    blocks = []
    block = []
    for line in lines:
        if line.startswith("|AddStaff"):
            if block:
                blocks.append(block)
            block = [line]
        elif line.startswith("|StaffProperties") or "|Label:" in line or "|Clef|Type:" in line:
            block.append(line)
    if block:
        blocks.append(block)
    return blocks

def infer_staff_roles(nwctxt: str) -> Tuple[Dict[str, str], str]:
    lines = nwctxt.splitlines()
    blocks = parse_staff_blocks(lines)
    role_map = {}
    staff_infos = []

    for i, block in enumerate(blocks):
        name = label = clef = ""
        brace = False
        for line in block:
            if "|AddStaff" in line:
                m = re.search(r'Name:"([^"]+)"', line)
                if m:
                    name = m.group(1)
            if "|Label:" in line:
                m = re.search(r'Label:"([^"]+)"', line)
                if m:
                    label = m.group(1)
            if "|Clef|Type:" in line:
                m = re.search(r'Type:([^\s|]+)', line)
                if m:
                    clef = m.group(1)
            if "|StaffProperties" in line and "Brace" in line:
                brace = True

        staff_id = label or name
        role = classify_by_keyword(staff_id, clef, brace)

        staff_infos.append({
            "index": i,
            "id": staff_id,
            "clef": clef,
            "brace": brace,
            "role": role,
        })

    # Step 1: Mark piano parts
    piano_ids = set()
    for info in staff_infos:
        if info["brace"]:
            if info["clef"].lower() == "treble":
                info["role"] = "Piano-RH"
            elif info["clef"].lower() == "bass":
                info["role"] = "Piano-LH"
            piano_ids.add(info["id"])
            role_map[info["id"]] = info["role"]

    # Step 2: Determine vocal parts (non-piano)
    non_piano = [s for s in staff_infos if s["id"] not in piano_ids]
    vocal_roles = []

    if len(non_piano) >= 4:
        vocal_roles = ["Soprano", "Alto", "Tenor", "Bass"]
    elif len(non_piano) == 2:
        vocal_roles = ["SA", "TB"]
    elif len(non_piano) > 0:
        vocal_roles = ["Soprano", "Alto", "Tenor", "Bass"][:len(non_piano)]

    for i, info in enumerate(non_piano):
        clef = info["clef"].lower()
        if i < len(vocal_roles):
            role = vocal_roles[i]
        else:
            # Infer from clef if extra vocal or unknown
            role = "Tenor" if clef == "bass" else "Alto" if clef == "treble" else "Soprano"
        info["role"] = role
        role_map[info["id"]] = role

    # Step 3: Remaining parts are instruments (VCF) or unknown
    for info in staff_infos:
        if info["id"] not in role_map:
            inferred = classify_by_keyword(info["id"], info["clef"], info["brace"])
            if inferred:
                role_map[info["id"]] = inferred

    # Determine postfix
    voices = {r for r in role_map.values() if r in {"Soprano", "Alto", "Tenor", "Bass", "SA", "TB"}}
    piano = {"Piano-RH", "Piano-LH"} & set(role_map.values())
    extras = {r for r in role_map.values() if r in VCF}

    postfix = ""
    if voices == {"Soprano", "Alto", "Tenor", "Bass"}:
        postfix = "SATB4_P" if piano else "SATB4"
    elif voices == {"SA", "TB"}:
        postfix = "SATB2_P" if piano else "SATB2"
    elif voices:
        postfix = "SATB_P" if piano else "SATB"
    elif piano:
        postfix = "P"

    if extras:
        postfix += "_" + "".join(sorted(e[0] for e in extras))

    return role_map, postfix



def apply_staff_name_replacements(content: str, mapping: Dict[str, str]) -> str:
    for orig, new in mapping.items():
        content = re.sub(rf'(\|AddStaff\|Name:)"{re.escape(orig)}"', rf'\1"{new}"', content)
        content = re.sub(rf'(\|Label:)"{re.escape(orig)}"', rf'\1"{new}"', content)
    return content


def rename_file_with_postfix(file: Path, postfix: str) -> Path:
    """
    Strip any existing '__POSTFIX' suffix and rename with new '__POSTFIX'.
    """
    # Strip any prior "__..." postfix
    base = re.sub(r'__[^.]+$', '', file.stem)
    new_name = f"{base}__{postfix}{file.suffix}"
    new_path = file.with_name(new_name)

    if new_path != file:
        file.rename(new_path)
        print(f"✅ Renamed: {file.name} → {new_path.name}")
    else:
        print(f"✅ Already correct name: {file.name}")
    return new_path


def process_folder(folder: Path):
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    files = list(folder.glob("*.nwctxt"))
    if not files:
        print(f"⚠️  No .nwctxt files found in {folder}")
        return

    for file in files:
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
            role_map, postfix = infer_staff_roles(content)
            if not role_map:
                print(f"⚠️  Skipping {file.name}: No roles inferred.")
                continue
            updated = apply_staff_name_replacements(content, role_map)
            file.write_text(updated, encoding="utf-8")
            if postfix:
                renamed = rename_file_with_postfix(file, postfix)
                print(f"✅ {file.name} → {renamed.name}")
            else:
                print(f"✅ Updated (no renaming): {file.name}")
        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nwctxt_rename.py <folder_path>")
        sys.exit(1)

    target = Path(sys.argv[1])
    process_folder(target)
