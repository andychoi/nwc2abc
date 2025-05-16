from music21 import instrument

def classify_parts(score):
    vocal_keywords = {"soprano", "alto", "tenor", "bass"}
    instrumental_parts = []
    vocal_parts = []

    for part in score.parts:
        name = part.partName.lower() if part.partName else ""
        inst = part.getInstrument(returnDefault=True)
        if any(voice in name for voice in vocal_keywords):
            vocal_parts.append(part)
        else:
            instrumental_parts.append(part)
    return vocal_parts, instrumental_parts
