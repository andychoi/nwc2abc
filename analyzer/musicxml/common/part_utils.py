from music21 import instrument

def classify_parts(score, exclude_piano=True):
    vocal_keywords = {"soprano", "alto", "tenor", "bass"}
    instrumental_parts = []
    vocal_parts = []

    for part in score.parts:
        name = part.partName.lower() if part.partName else ""
        inst = part.getInstrument(returnDefault=True)
        is_piano = "piano" in name or "piano" in str(inst).lower()

        if any(voice in name for voice in vocal_keywords):
            vocal_parts.append(part)
        elif not (exclude_piano and is_piano):
            instrumental_parts.append(part)
            
    return vocal_parts, instrumental_parts

