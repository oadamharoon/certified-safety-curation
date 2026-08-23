"""Scan the tree for the specific defect classes that have bitten this project.

Each check exists because the bug it looks for actually shipped and silently
corrupted results. None of them raise at runtime, which is what makes them
dangerous.

  1 double-escaped regex   r"(\\d+)" inside a raw string is a literal
                           backslash-d. Five harvester patterns had this and
                           matched nothing for weeks.
  2 missing SEED_OVERRIDE  a training script that seeds but never reads the
                           override runs every "seed" identically.
  3 unexported bash fn     a function called inside xargs/parallel subshells
                           without export -f is undefined there; the call fails
                           silently while the surrounding logic continues.
  4 ephemeral marker dir   done-markers under /tmp or a session scratchpad
                           cannot survive, so reruns cannot resume.
"""
import os
import re
import sys

ROOT = "/home/omniverse/workspace/safevlmcpl"
DIRS = ["certified-safety-curation/pipeline", "certified-safety-curation/analysis",
        "certified-safety-curation/experiments", "iclr2027/scripts",
        "runs/scripts", "vlm-with-cpl/new_data/scripts"]
SKIP = ("legacy/", "_quarantine", "/.git/")

BAD_ESCAPE = re.compile(r'r"[^"]*\\\\[dwsSWDb+.*][^"]*"|r\'[^\']*\\\\[dwsSWDb+.*][^\']*\'')


def seeded_by_runner():
    """Basenames of python scripts launched with SEED_OVERRIDE set."""
    out = set()
    for d in DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if not fn.endswith(".sh"):
                continue
            src = open(os.path.join(p, fn), errors="ignore").read()
            for blk in re.findall(r"SEED_OVERRIDE=[^\n]*?python\s+\S*?([\w.]+\.py)",
                                  src, re.S):
                out.add(os.path.basename(blk))
    return out


def files(ext):
    for d in DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if fn.endswith(ext) and not any(s in fn for s in SKIP):
                yield os.path.join(p, fn)


SEEDED_BY_RUNNER = set()


def main():
    global SEEDED_BY_RUNNER
    SEEDED_BY_RUNNER = seeded_by_runner()
    hits = {1: [], 2: [], 3: [], 4: []}

    for f in files(".py"):
        src = open(f, errors="ignore").read()
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if BAD_ESCAPE.search(code) and ("=" in code or "return" in code):
                hits[1].append(f"{f}:{i}  {line.strip()[:88]}")
        # only a defect if a runner actually passes SEED_OVERRIDE to it
        if ("set_seed(" in src and "SEED_OVERRIDE" not in src
                and os.path.basename(f) in SEEDED_BY_RUNNER):
            hits[2].append(f"{f}  (launched with SEED_OVERRIDE, never reads it)")

    for f in files(".sh"):
        src = open(f, errors="ignore").read()
        defined = set(re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*\(\)\s*\{", src, re.M))
        exported = set()
        for m in re.findall(r"export -f ([^\n]+)", src):
            exported |= {x.strip() for x in re.split(r"[;\s]+", m) if x.strip()}
        # only matters if the script actually dispatches through a subshell
        if re.search(r"xargs .*bash -c|parallel ", src):
            for fn in sorted(defined - exported):
                if re.search(rf"\b{fn}\b", src.split("export -f")[-1] if "export -f" in src else src):
                    hits[3].append(f"{f}  {fn}() defined, used in subshell, never export -f'd")
        for m in re.finditer(r"^\s*LOGDIR=(\S+)", src, re.M):
            if "/tmp" in m.group(1) or "scratchpad" in m.group(1):
                hits[4].append(f"{f}  LOGDIR={m.group(1)}")

    titles = {1: "double-escaped regex in raw string",
              2: "training script ignoring SEED_OVERRIDE",
              3: "bash function used in subshell but not exported",
              4: "done-markers in an ephemeral directory"}
    bad = 0
    for k in (1, 2, 3, 4):
        print(f"[{k}] {titles[k]}: {len(hits[k])}")
        for h in hits[k]:
            print(f"      {h.replace(ROOT + '/', '')}")
        bad += len(hits[k])
    print()
    print("CLEAN" if bad == 0 else f"{bad} issue(s) found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
