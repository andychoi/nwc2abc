# common/html_report.py

from common.abc_utils import pitch_to_abc
from music21 import roman, pitch as m21pitch, key as m21key
import html

default_key = m21key.Key('C')

# HTML template with tooltip support for ABC hover
html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Harmony Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/abcjs@6.4.0/dist/abcjs-basic-min.js"></script>
    <style>
        .ok {{ background-color: #e9fce9; }}
        .voicing {{ background-color: #fff3cd; color: #856404; }}
        .dissonance {{ background-color: #fdecea; color: #b94a48; }}
        .doubling {{ background-color: #d1ecf1; color: #0c5460; }}
        th, td {{ text-align: center; vertical-align: middle; }}
        .note-issue {{ margin-bottom: 0.5em; }}
        .note-issue b {{ display: block; }}
        .note-issue small {{ font-style: italic; }}
        .abc-hover {{ cursor: help; text-decoration: dotted underline; }}
        .abc-tooltip {{
            position: absolute;
            background: #fff;
            border: 1px solid #ccc;
            padding: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            z-index: 10000;
        }}
    </style>
</head>
<body class="container my-4">
    <h1 class="mb-4">Harmony Analysis Report</h1>
    <table class="table table-bordered table-sm">
        <thead class="table-light">
            <tr>
                <th>Measure</th><th>Chord</th>{columns}
            </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
    </table>
    <h2 class="mt-5">ABC Preview</h2>
    <div id="abc"></div>
    <script>
        // Render full ABC preview at bottom
        window.onload = function() {{
            const abc = `{abc_preview}`;
            if (window.ABCJS && ABCJS.renderAbc) {{
                ABCJS.renderAbc("abc", abc);
            }}
        }};
    </script>
    <script>
        // Tooltip behavior for chord hover
        document.addEventListener('DOMContentLoaded', () => {{
            document.querySelectorAll('.abc-hover').forEach(el => {{
                el.addEventListener('mouseenter', () => {{
                    const abc = el.getAttribute('data-abc');
                    if (!abc) return;
                    const id = 'tt-' + Math.random().toString(36).substr(2,9);
                    const tt = document.createElement('div');
                    tt.id = id;
                    tt.className = 'abc-tooltip';
                    document.body.appendChild(tt);
                    const rect = el.getBoundingClientRect();
                    tt.style.left = `${{rect.left + window.scrollX}}px`;
                    tt.style.top = `${{rect.bottom + window.scrollY}}px`;
                    ABCJS.renderAbc(id, abc);
                    el._tooltipId = id;
                }});
                el.addEventListener('mouseleave', () => {{
                    if (el._tooltipId) {{
                        const tt = document.getElementById(el._tooltipId);
                        if (tt) tt.remove();
                        el._tooltipId = null;
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>"""


def classify_issue(issue_text):
    it = issue_text.lower()
    if "dissonance" in it:
        return "dissonance"
    if "doubled" in it:
        return "doubling"
    return "voicing"


def recommend_fix(issue_text):
    it = issue_text.lower()
    if "dissonance with " in it:
        target = issue_text.split("dissonance with ")[1]
        p = m21pitch.Pitch(target)
        abc_orig = pitch_to_abc(p)
        candidates = [default_key.pitchFromDegree(d) for d in range(1, 8)]
        for sp in candidates:
            sp.octave = p.octave
        candidates.sort(key=lambda sp: abs(sp.midi - p.midi))
        best = candidates[1] if candidates and candidates[0].midi == p.midi else candidates[0]
        abc_res = pitch_to_abc(best)
        mode = '' if default_key.mode == 'major' else 'm'
        return f"Resolve dissonance: change {abc_orig} → {abc_res} ({default_key.tonic.name}{mode} scale)"
    if "spacing" in it:
        return "Reduce to within an octave: e.g., lower upper voice by 8ve (ABC: A→a)."
    if "crossing" in it:
        return "Maintain voice order: transpose the lower voice down (e.g., ABC: A,)."
    if "parallel p5" in it:
        return "Break parallel 5th: insert passing tone in one voice (e.g., z in ABC)."
    if "parallel p8" in it:
        return "Avoid parallel 8ve: use contrary motion (e.g., step in ABC: c d c)."
    if "doubled in " in it:
        part = issue_text.split("doubled in ")[1]
        return f"Drop duplication in {part} or vary voicing—e.g., use ABC chord [CE] instead of [CC]."
    if "prog" in it:
        tonic = pitch_to_abc(default_key.tonic)
        dominant = pitch_to_abc(default_key.pitchFromDegree(5))
        return f"V→I in ABC: {dominant} {dominant} {dominant}| {tonic} {tonic} {tonic}"
    return "Review harmonic context."


def render_html_report(issues_by_measure, part_names, output_path, chords_by_measure=None, abc_key=None):
    # Build columns and rows
    columns_html = ''.join(f'<th>{part}</th>' for part in part_names)
    rows_html = []
    all_abc = []

    for m in sorted(issues_by_measure.keys()):
        chords = chords_by_measure.get(m, []) if chords_by_measure else []
        if isinstance(chords, str): chords = [chords]
        labels = ", ".join(chords)
        data_abc = ""
        if abc_key and chords:
            parts = []
            for rn in chords:
                try:
                    obj = roman.RomanNumeral(rn, abc_key)
                    pitches = [pitch_to_abc(p, abc_key) for p in obj.pitches]
                    parts.append(f'"{rn}" [{" ".join(pitches)}]')
                except:
                    parts.append(f'"{rn}" z')
            data_abc = " ".join(parts)
            all_abc.append(data_abc)
        hover_div = f'<div class="abc-hover" data-abc="{html.escape(data_abc)}">{labels}</div>' if labels else ''

        cells = [f'<td>{m}</td>', f'<td>{hover_div}</td>']
        for part in part_names:
            issues = issues_by_measure.get(m, {}).get(part, [])
            if not issues:
                cells.append('<td class="ok">✓</td>')
            else:
                inner = ''.join(f'<div class="note-issue {classify_issue(i)}"><b>{i}</b><small>{recommend_fix(i)}</small></div>' for i in issues)
                cells.append(f'<td>{inner}</td>')
        rows_html.append('<tr>' + ''.join(cells) + '</tr>')

    full_html = html_template.format(
        columns=columns_html,
        rows="\n".join(rows_html),
        abc_preview="\n".join(all_abc)
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)


def render_style_report(issues_by_measure, part_names, output_path, chords_by_measure=None, abc_key=None):
    render_html_report(issues_by_measure, part_names, output_path, chords_by_measure, abc_key)
