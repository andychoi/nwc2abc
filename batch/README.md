# NWC to ABC Notation Conversion Toolkit

This toolkit provides a reliable multi-stage pipeline to convert Noteworthy Composer (`.nwc`) files into ABC notation (`.abc`) via `.nwctxt` and `.musicxml`. It also includes tools for recovering broken Korean text (mojibake) caused by encoding issues.

---

## 📂 Conversion Pipeline Overview

```
.nwc → .nwctxt → .musicxml → .abc
          │         │         └─ optional & repeatable
          │         └─ organized into folders by composer
          └─ fix Korean mojibake in metadata & lyrics if needed
```

---

## 🔧 Requirements

### Software

* ✅ Noteworthy Composer 2 CLI (`nwc2.exe`)
* ✅ Java 8+ with `nwc2musicxml.jar`
* ✅ Python 3.8+ with `music21`, and custom `nwc2abc` module
* ✅ PowerShell 5+ or newer (for .ps1 scripts)

---

## 📁 Script Overview

| Script Name             | Description                                                           |
| ----------------------- | --------------------------------------------------------------------- |
| `nwc2nwctxt.ps1`        | Recursively converts `.nwc` → `.nwctxt` using `nwc2.exe`              |
| `fix-korean.py`         | Fixes corrupted Hangul in `.nwctxt` files caused by encoding mismatch |
| `nwctxt2musicxml.ps1`   | Converts `.nwctxt` → `.musicxml` via `nwc2musicxml.jar` (Java)        |
| `musicxml-organize.ps1` | Moves `.musicxml` files into folders by composer metadata             |
| `musicxml2abc.ps1`      | Recursively converts `.musicxml` → `.abc` using your Python module    |

---

## 🚀 How to Use

### 1️⃣ Convert `.nwc` → `.nwctxt`

```powershell
.\nwc2nwctxt.ps1 "nwcoriginal" "nwctxt"
```

👉 Add `-Force` to overwrite all:

```powershell
.\nwc2nwctxt.ps1 "nwcoriginal" "nwctxt" -Force
```

---

### 2️⃣ (Optional) Fix Korean mojibake in `.nwctxt`

```bash
python fix-korean.py nwctxt
```

📂 Creates `nwctxt-utf8` by default (or use your own target folder):

```bash
python fix-korean.py nwctxt fixed-nwctxt --force
```

---

### 3️⃣ Convert `.nwctxt` → `.musicxml`

```powershell
.\nwctxt2musicxml.ps1 "nwctxt-utf8"
```

👉 Add `-Force` to overwrite outdated `.musicxml`:

```powershell
.\nwctxt2musicxml.ps1 "nwctxt-utf8" -Force
```

---

### 4️⃣ Organize `.musicxml` by composer

```powershell
.\musicxml-organize.ps1 "musicxml"
```

📁 Output will be grouped into:

```
musicxml/
├── Bach/
│   └── song.musicxml
├── Mozart/
│   └── aria.musicxml
└── unknown/
    └── mystery.musicxml
```

---

### 5️⃣ (Optional) Convert `.musicxml` → `.abc`

```powershell
.\musicxml2abc.ps1 "musicxml"
```

👉 Add `-Force` to reprocess all `.abc`:

```powershell
.\musicxml2abc.ps1 "musicxml" -Force
```

---

## 🧠 Smart Behavior

| Script                | Skips Up-to-Date Files | Supports `--force` or `-Force` |
| --------------------- | ---------------------- | ------------------------------ |
| `nwc2nwctxt.ps1`      | ✅                      | ✅                              |
| `fix-korean.py`       | ✅                      | ✅                              |
| `nwctxt2musicxml.ps1` | ✅                      | ✅                              |
| `musicxml2abc.ps1`    | ✅                      | ✅                              |

---

## 🧪 Preview ABC Output

Test your `.abc` files using:

🔗 [Michael Eskin’s ABC Tools](https://michaeleskin.com/abctools/abctools.html)

---

## 📌 Extras

* Scripts preserve folder structure automatically
* You can chain these in a custom `convert-all.ps1`
* Hangul font names like `"굴림체"` are recovered or replaced if corrupted
* `fix-korean.py` detects mojibake and only fixes broken fields
