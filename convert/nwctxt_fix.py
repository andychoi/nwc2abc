
#!/usr/bin/env python3
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# General MIDI Patch Names
GM_PATCH_NAMES = {
    0: "Acoustic Grand Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    13: "Xylophone", 24: "Acoustic Guitar (nylon)", 32: "Acoustic Bass",
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    46: "Orchestral Harp", 48: "String Ensemble 1", 49: "String Ensemble 2",
    50: "SynthStrings 1", 51: "SynthStrings 2", 52: "Choir Aahs", 53: "Voice Oohs",
    54: "Synth Voice", 56: "Trumpet", 64: "Soprano Sax", 65: "Alto Sax",
    66: "Tenor Sax", 67: "Baritone Sax", 68: "Oboe", 71: "Clarinet",
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute"
}

ALT_PATCHES = {
    "alt1": {
        "Soprano 1": 52, "Soprano 2": 53, "Soprano": 52,
        "Alto": 41,
        "Tenor": 66,
        "Bass": 32,
    },
    "alt2": {
        "Soprano 1": 72, "Soprano 2": 73, "Soprano": 72,
        "Alto": 65,
        "Tenor": 42,
        "Bass": 67,
    }
}

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

def parse_staff_blocks(lines: List[str]) -> List[List[str]]:
    blocks, current = [], []
    for ln in lines:
        if ln.startswith("|AddStaff"):
            if current:
                blocks.append(current)
            current = [ln]
        elif current and (ln.startswith("|StaffProperties") or ln.startswith("|Label:") or "|Clef|Type:" in ln or ln.startswith("|StaffInstrument")):
            current.append(ln)
    if current:
        blocks.append(current)
    return blocks

def classify_by_keyword(name: str, clef: str = "", brace: bool = False) -> str:
    key = normalize(name)
    for k, role in ROLE_KEYWORDS.items():
        if key == k or re.search(rf"\b{k}\b", key):
            return role
    if brace:
        return "Piano-RH"
    return ""

# ... [keep original import and global constants unchanged] ...

def infer_staff_roles(content: str, return_details=False) -> Tuple[Dict[int, str], str, List[str], List[str], List[str], List[str]]:
    lines = content.splitlines()
    blocks = parse_staff_blocks(lines)
    n = len(blocks)

    names, labels, clefs, instruments, braces = [], [], [], [], []
    for block in blocks:
        name = label = clef = patch = ""
        brace = False
        for ln in block:
            if ln.startswith("|AddStaff"):
                m = re.search(r'Name:"([^"]+)"', ln)
                if m: name = m.group(1)
                m2 = re.search(r'Label:"([^"]+)"', ln)
                if m2: label = m2.group(1)
            elif ln.startswith("|Label:"):
                m = re.search(r'"([^"]+)"', ln)
                if m: label = m.group(1)
            elif "|Clef|Type:" in ln:
                m = re.search(r'Type:([^\s|]+)', ln)
                if m: clef = m.group(1)
            elif ln.startswith("|StaffProperties") and "Brace" in ln:
                brace = True
            elif ln.startswith("|StaffInstrument"):
                m = re.search(r'Patch:(\d+)', ln)
                if m: patch = f"Patch:{m.group(1)}"
        names.append(name)
        labels.append(label)
        clefs.append(clef)
        instruments.append(patch or "-")
        braces.append(brace)

    piano_rh_idxs = [i for i, br in enumerate(braces) if br]
    piano_idxs = set()
    for i in piano_rh_idxs:
        piano_idxs.add(i)
        if i + 1 < n:
            piano_idxs.add(i + 1)

    voice_idxs = [i for i in range(n) if i not in piano_idxs]
    role_map: Dict[int, str] = {}

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
        role_map[idx] = treble_roles[i] if i < len(treble_roles) else "Soprano"

    for idx in bass_idxs:
        role_map[idx] = "Tenor" if "Tenor" not in role_map.values() else "Bass"

    for i in sorted(piano_idxs):
        role_map[i] = "Piano-RH" if i in piano_rh_idxs else "Piano-LH"

    for i in range(n):
        if i not in role_map:
            inferred = classify_by_keyword(names[i], clefs[i], braces[i])
            role_map[i] = inferred or "Extra"

    all_roles = set(role_map.values())
    base_roles = {r.split()[0] for r in all_roles}
    voices_set = base_roles & {"Soprano", "Alto", "Tenor", "Bass", "SA", "TB"}
    piano_set = all_roles & {"Piano-RH", "Piano-LH"}
    extras = base_roles & VCF

    postfix = ""
    
    if {"Soprano", "Alto", "Tenor", "Bass"} <= voices_set:
        postfix = "SATB_P" if piano_set else "SATB"
    elif {"SA", "TB"} <= voices_set:
        postfix = "SA_TB_P" if piano_set else "SA_TB"
    elif voices_set:
        if len(voices_set) >= 2:
            postfix = "Choral_P" if piano_set else "Choral"
        else:
            postfix = list(voices_set)[0]
            if piano_set:
                postfix += "_P"
    elif piano_set:
        postfix = "P"

    if extras:
        postfix += "_" + "".join(sorted(e[0] for e in extras))

    return role_map, postfix, names, labels, clefs, instruments

def apply_updates(content: str, role_map: Dict[int, str], alt: str = None) -> str:
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
                    ln = ln.replace("|AddStaff", f'|AddStaff|Label:"{abbr}"')
        elif ln.startswith("|Label:"):
            abbr = abbreviate_label(role_map.get(staff_idx, ""))
            ln = f'|Label:"{abbr}"'
        elif ln.startswith("|StaffInstrument") and alt:
            role = role_map.get(staff_idx)
            patch_map = ALT_PATCHES.get(alt, {})
            if role in patch_map:
                new_patch = patch_map[role]
                ln = re.sub(r'Patch:\d+', f'Patch:{new_patch}', ln)
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
        print(f"✅ Name unchanged: {file.name}")
    return new_path

def process_folder(folder: Path, rename: bool = True, test_mode: bool = False, alt_patch: str = None):
    if not folder.exists():
        print(f"📁 Creating missing folder: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
    files = list(folder.glob("*.nwctxt"))
    if not files:
        print(f"⚠️  No .nwctxt files found in {folder}")
        return
    for file in files:
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
            role_map, postfix, names, labels, clefs, instruments = infer_staff_roles(content, return_details=True)

            if test_mode:
                print(f"\n🔍 Simulating: {file.name}")

                for i, role in role_map.items():
                    name = names[i]
                    label = labels[i]
                    clef = clefs[i]
                    instr = instruments[i]
                    
                    # Parse original patch number
                    patch_num = int(instr.split(":")[1]) if instr.startswith("Patch:") else None
                    patch_name = GM_PATCH_NAMES.get(patch_num, "Unknown") if patch_num is not None else "-"

                    # Get target patch (if alt patching is active)
                    target_patch = None
                    target_patch_name = "-"
                    if alt_patch and role in ALT_PATCHES.get(alt_patch, {}):
                        target_patch = ALT_PATCHES[alt_patch][role]
                        target_patch_name = GM_PATCH_NAMES.get(target_patch, "Unknown")

                    # Show both original and converted MIDI instrument names
                    if target_patch is not None and patch_num != target_patch:
                        patch_display = f"{instr} ({patch_name}) → Patch:{target_patch} ({target_patch_name})"
                    else:
                        patch_display = f"{instr} ({patch_name})"

                    print(f"  STAFF {i}: {name} / {label} / {clef} / {patch_display} → {role} ({abbreviate_label(role)})")                
                continue

            updated = apply_updates(content, role_map, alt_patch)
            file.write_text(updated, encoding="utf-8")

            if rename and postfix:
                rename_file_with_postfix(file, postfix)
            else:
                print(f"✅ Updated (no renaming): {file.name}")
        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nwctxt_fix.py <folder_path> [--no-rename] [--test] [--alt=alt1|alt2]")
        sys.exit(1)
    target = Path(sys.argv[1])
    do_rename = "--no-rename" not in sys.argv
    is_test = "--test" in sys.argv
    alt_patch = "alt1" 
    for arg in sys.argv[2:]:
        if arg.startswith("--alt="):
            alt_patch = arg.split("=", 1)[1]
    process_folder(target, rename=do_rename, test_mode=is_test, alt_patch=alt_patch)
