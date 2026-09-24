#!/bin/bash
# Build the anonymous code archive for the ICLR supplementary upload.
#
# It is the repository's tracked codebase (`git ls-files`) with the authorship block stripped
# from README.md and the author block redacted from the paper source, so the archive still
# runs the full audit (the orphan guard reads paper.tex) without naming anybody. A scrub step
# fails the build if any author name, institution, e-mail or account string survives anywhere.
#
# Usage: bash tools/make_anonymous_zip.sh [outdir]     default outdir: the repository's parent
set -euo pipefail
ROOT="${CSC_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${1:-$(dirname "$ROOT")}"
NAME="certified-safety-curation-code"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$NAME"
cd "$ROOT"
command -v git >/dev/null || { echo "git is required"; exit 1; }
git rev-parse --verify HEAD >/dev/null 2>&1 || { echo "commit the repository first"; exit 1; }

mkdir -p "$DEST"
git ls-files -z | while IFS= read -r -d '' f; do
  case "$f" in
    tools/make_anonymous_zip.sh) continue ;;   # this builder greps for the names
    README.md|paper/paper.tex) continue ;;     # both handled below
  esac
  mkdir -p "$DEST/$(dirname "$f")"; cp "$f" "$DEST/$f"
done

# README without the authorship block
python - "$ROOT/README.md" "$DEST/README.md" <<'EOF'
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
t = re.sub(r"<!-- AUTHORSHIP:.*?<!-- END AUTHORSHIP -->\n+", "", t, flags=re.S)
open(sys.argv[2], "w", encoding="utf-8").write(t)
EOF

# paper source with the author block and the corresponding-author footnote redacted. The
# archive keeps the source because the audit's orphan guard reads it: a check that guards a
# value no longer in the text has to fail here too, or the archive's gate is weaker than ours.
mkdir -p "$DEST/paper"
python - "$ROOT/paper/paper.tex" "$DEST/paper/paper.tex" <<'EOF'
import sys

t = open(sys.argv[1], encoding="utf-8").read()
# Brace-matched, not a regex: the author block nests \thanks{\texttt{...}} and closes with
# "}}" on its own line, which every simple pattern for it gets wrong in one direction.
i = t.index("\\author{")
j = i + len("\\author{")
depth = 1
while depth:
    if t[j] == "{":
        depth += 1
    elif t[j] == "}":
        depth -= 1
    j += 1
t = t[:i] + "\\author{Anonymous authors \\\\\nPaper under double-blind review}" + t[j:]
open(sys.argv[2], "w", encoding="utf-8").write(t)
EOF

# Scrub: nothing identifying may survive anywhere in the archive.
python - "$DEST" <<'EOF' || exit 1
import os, re, sys
stage = sys.argv[1]
PAT = re.compile(r"haroon|fleming|iastate|iowa state|ames,\s*ia|oadamharoon|aharoon", re.I)
bad, n = [], 0
for root, _, files in os.walk(stage):
    for f in files:
        p = os.path.join(root, f); n += 1
        try: txt = open(p, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        bad += [(os.path.relpath(p, stage), m.group(0)) for m in PAT.finditer(txt)]
if bad:
    print("SCRUB FAILED -- identifying strings in the archive:")
    for rel, s in bad[:25]: print("   ", rel, "->", s)
    sys.exit(1)
print("scrub clean: no author, institution, e-mail or account string in %d files" % n)
EOF

mkdir -p "$OUT"; rm -f "$OUT/$NAME.zip"
( cd "$STAGE" && zip -qr "$OUT/$NAME.zip" "$NAME" )
echo "archive: $OUT/$NAME.zip  ($(du -sh "$OUT/$NAME.zip" | cut -f1), $(find "$DEST" -type f | wc -l) files)"
