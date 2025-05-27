#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Role keyword mappings
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

def classify_by_keyword(name: str, clef: str = "", brace: bool = False) -> str:
    key = normalize(name)
    for k, role in ROLE_KEYWORDS.items():
        if key == k or re.search(rf"\b{k}\b", key):
            return role
    if brace:
        return "Piano-RH"
    return ""

def parse_staff_blocks(lines: List[str]) -> List[List[str]]:
    blocks, current = [], []
    for ln in lines:
        if ln.startswith("|AddStaff"):
            if current:
                blocks.append(current)
            current = [ln]
        elif current and (ln.startswith("|StaffProperties") or ln.startswith("|Label:") or "|Clef|Type:" in ln):
            current.append(ln)
    if current:
        blocks.append(current)
    return blocks

def infer_staff_roles(content: str) -> Tuple[Dict[int, str], str]:
    lines = content.splitlines()
    blocks = parse_staff_blocks(lines)
    n = len(blocks)

    names, clefs, braces = [], [], []
    for block in blocks:
        nm = clef = ""
        br = False
        for ln in block:
            if ln.startswith("|AddStaff"):
                m = re.search(r'Name:"([^"]+)"', ln)
                if m: nm = m.group(1)
            if ln.startswith("|StaffProperties") and "Brace" in ln:
                br = True
            if "|Clef|Type:" in ln:
                m = re.search(r'Type:([^\s|]+)', ln)
                if m: clef = m.group(1)
        names.append(nm)
        clefs.append(clef)
        braces.append(br)

    # Detect piano indices using brace (RH) and next index (LH)
    piano_rh_idxs = [i for i, br in enumerate(braces) if br]
    piano_idxs = set()
    for i in piano_rh_idxs:
        piano_idxs.add(i)
        if i + 1 < n:
            piano_idxs.add(i + 1)

    # Everything else = voice blocks
    voice_idxs = [i for i in range(n) if i not in piano_idxs]
    role_map: Dict[int, str] = {}
    counts: Dict[str, int] = {}

    # Voice logic based on clef
    treble_idxs = [i for i in voice_idxs if clefs[i].lower() == "treble"]
    bass_idxs = [i for i in voice_idxs if clefs[i].lower() == "bass"]

    treble_roles = []
    if len(treble_idxs) == 1:
        treble_roles = ["Soprano"]
    elif len(treble_idxs) == 2:
        treble_roles = ["Soprano", "Alto"]
    elif len(treble_idxs) == 3:
        treble_roles = ["Soprano 1", "Soprano 2", "Alto"]
    elif len(treble_idxs) >= 4:
        treble_roles = ["Soprano 1", "Soprano 2", "Alto", "Alto 2"]

    for i, idx in enumerate(treble_idxs):
        role = treble_roles[i] if i < len(treble_roles) else "Soprano"
        role_map[idx] = role

    for idx in bass_idxs:
        # Assign Tenor first, then Bass
        if "Tenor" not in role_map.values():
            role_map[idx] = "Tenor"
        else:
            role_map[idx] = "Bass"

    # Piano roles
    for i in sorted(piano_idxs):
        if i in piano_rh_idxs:
            role_map[i] = "Piano-RH"
        else:
            role_map[i] = "Piano-LH"

    # Remaining roles by keyword or Extra
    for i in range(n):
        if i not in role_map:
            inferred = classify_by_keyword(names[i], clefs[i], braces[i])
            role_map[i] = inferred or "Extra"

    # Build postfix
    all_roles = set(role_map.values())
    base_roles = {r.split()[0] for r in all_roles}
    voices_set = base_roles & {"Soprano", "Alto", "Tenor", "Bass", "SA", "TB"}
    piano_set = all_roles & {"Piano-RH", "Piano-LH"}
    extras = base_roles & VCF

    postfix = ""
    if {"Soprano", "Alto", "Tenor", "Bass"} <= voices_set:
        postfix = "SATB4_P" if piano_set else "SATB4"
    elif {"SA", "TB"} <= voices_set:
        postfix = "SATB2_P" if piano_set else "SATB2"
    elif voices_set:
        postfix = "SATB_P" if piano_set else "SATB"
    elif piano_set:
        postfix = "P"
    if extras:
        postfix += "_" + "".join(sorted(e[0] for e in extras))

    return role_map, postfix

def abbreviate_label(role: str) -> str:
    if role.startswith("Soprano"):
        parts = role.split()
        return "S" if len(parts) == 1 else f"S{parts[1]}"
    if role == "Alto": return "A"
    if role == "Alto 2": return "A2"
    if role == "Tenor": return "T"
    if role == "Bass": return "B"
    if role in {"SA", "TB"}: return role
    if role == "Piano-RH": return "PRH"
    if role == "Piano-LH": return "PLH"
    return role[:3]

def apply_staff_name_replacements(content: str, role_map: Dict[int, str]) -> str:
    lines = content.splitlines()
    new_lines: List[str] = []
    staff_idx = -1
    for ln in lines:
        if ln.startswith("|AddStaff"):
            staff_idx += 1
            role = role_map.get(staff_idx)
            if role:
                abbr = abbreviate_label(role)
                ln = re.sub(r'(\|AddStaff\|Name:)"[^"]+"', rf'\1"{role}"', ln)
                if "|Label:" in ln:
                    ln = re.sub(r'(\|Label:)"[^"]+"', rf'\1"{abbr}"', ln)
                else:
                    ln = ln.replace("|AddStaff", f"|AddStaff|Label:\"{abbr}\"")
        elif ln.startswith("|Label:"):
            abbr = abbreviate_label(role_map.get(staff_idx, ""))
            ln = f'|Label:"{abbr}"'
        new_lines.append(ln)
    return "\n".join(new_lines)

def rename_file_with_postfix(file: Path, postfix: str) -> Path:
    base = re.sub(r'__[^.]+$', '', file.stem)
    new_name = f"{base}__{postfix}{file.suffix}"
    new_path = file.with_name(new_name)
    if new_path != file:
        file.rename(new_path)
        print(f"✅ Renamed: {file.name} → {new_path.name}")
    else:
        print(f"✅ Already correct name: {file.name}")
    return new_path

def process_folder(folder: Path, rename: bool = True):
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
            updated = apply_staff_name_replacements(content, role_map)
            file.write_text(updated, encoding="utf-8")
            if rename and postfix:
                rename_file_with_postfix(file, postfix)
            else:
                print(f"✅ Updated (no renaming): {file.name}")
        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nwctxt_rename.py <folder_path> [--no-rename]")
        sys.exit(1)
    target = Path(sys.argv[1])
    do_rename = "--no-rename" not in sys.argv
    process_folder(target, rename=do_rename)