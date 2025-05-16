# common/html_report.py

from common.abc_utils import pitch_to_abc
from music21 import pitch as m21pitch, key as m21key

# You can change this to whatever key the piece is in:
default_key = m21key.Key('C')

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Harmony Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .ok {{ background-color: #e9fce9; }}
        .voicing {{ background-color: #fff3cd; color: #856404; }}
        .dissonance {{ background-color: #fdecea; color: #b94a48; }}
        .doubling {{ background-color: #d1ecf1; color: #0c5460; }}
        th, td {{ text-align: center; vertical-align: middle; }}
        .note-issue {{ margin-bottom: 0.5em; }}
        .note-issue b {{ display: block; }}
        .note-issue small {{ font-style: italic; }}
    </style>
</head>
<body class="container my-4">
    <h1 class="mb-4">Harmony Analysis Report</h1>
    <table class="table table-bordered table-sm">
        <thead class="table-light">
            <tr>
                <th>Measure</th>
                {columns}
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>
"""

def classify_issue(issue_text):
    it = issue_text.lower()
    if "dissonance" in it:
        return "dissonance"
    if "doubled" in it:
        return "doubling"
    return "voicing"

def recommend_fix(issue_text):
    it = issue_text.lower()
    # 1) Dissonance resolution – pick nearest diatonic step in default_key
    if "dissonance with " in it:
        target = issue_text.split("dissonance with ")[1]
        p = m21pitch.Pitch(target)
        abc_orig = pitch_to_abc(p)
        # build diatonic candidates for same octave
        candidates = []
        for deg in range(1, 8):
            sp = default_key.pitchFromDegree(deg)
            sp.octave = p.octave
            candidates.append(sp)
        # sort by proximity
        candidates.sort(key=lambda sp: abs(sp.midi - p.midi))
        # avoid the same note
        best = candidates[1] if candidates and candidates[0].midi == p.midi else candidates[0]
        abc_res = pitch_to_abc(best)
        mode = '' if default_key.mode=='major' else 'm'
        return (f"Resolve dissonance: change {abc_orig} → {abc_res} "
                f"({default_key.tonic.name}{mode} scale)")

    # 2) Spacing
    if "spacing" in it:
        return "Reduce to within an octave: e.g., lower upper voice by 8ve (ABC: A→a)."

    # 3) Crossing
    if "crossing" in it:
        return "Maintain voice order: transpose the lower voice down (e.g., ABC: A,)."

    # 4) Parallel fifth
    if "parallel p5" in it:
        return "Break parallel 5th: insert passing tone in one voice (e.g., z in ABC)."

    # 5) Parallel octave
    if "parallel p8" in it:
        return "Avoid parallel 8ve: use contrary motion (e.g., step in ABC: c d c)."

    # 6) Doubling
    if "doubled in " in it:
        part = issue_text.split("doubled in ")[1]
        return (f"Drop duplication in {part} or vary voicing—"
                f"e.g., use ABC chord [CE] instead of [CC].")

    # 7) Progression
    if "prog" in it:
        tonic = pitch_to_abc(default_key.tonic)
        dominant = pitch_to_abc(default_key.pitchFromDegree(5))
        return (f"V→I in ABC: {dominant} {dominant} {dominant}| {tonic} {tonic} {tonic}")

    return "Review harmonic context."

def render_html_report(issues_by_measure, part_names, output_path):
    # Build table header
    columns_html = ''.join(f'<th>{part}</th>' for part in part_names)

    rows = []
    for m in sorted(issues_by_measure.keys()):
        cells = [f'<td>{m}</td>']
        for part in part_names:
            issues = issues_by_measure.get(m, {}).get(part, [])
            if not issues:
                cells.append('<td class="ok">✓</td>')
            else:
                cell_html = []
                for issue in issues:
                    css = classify_issue(issue)
                    fix = recommend_fix(issue)
                    cell_html.append(
                        f'<div class="note-issue {css}">'
                        f'<b>{issue}</b>'
                        f'<small>{fix}</small>'
                        f'</div>'
                    )
                cells.append(f'<td>{"".join(cell_html)}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    html = html_template.format(columns=columns_html, rows="\n".join(rows))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
