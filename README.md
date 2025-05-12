# nwc2abc

Convert Noteworthy Composer (NWCtxt) files to ABC notation (optimized for GenAI experiments).

## Features
- Converts NWC via musescore/nwc2musicxml
- Outputs ABC at 3 levels (`raw`, `medium`, `simple`)
- Ready for AI arrangement tasks

## Install

```bash
git clone ...
cd nwc2abc
pip install .
```

## Usage

```python
from nwc2abc import nwc_to_simplified_abc
abc = nwc_to_simplified_abc("your_score.nwc.txt", simplicity_level="medium")
print(abc)
```

or CLI:

```bash
nwc2abc-cli your_score.nwc.txt --level simple
```

## Requirements
- Python 3.8+
- Java (for `nwc2musicxml.jar` if using local mode)

## License
MIT