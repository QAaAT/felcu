"""Check the supported Python and dependency versions against SPEC 0, and update them.

SPEC 0 (https://scientific-python.org/specs/spec-0000/) supports each Python
feature release for 3 years and each release of a core package for 2 years.
This compares those windows with the floors in pyproject.toml and the Python
versions tested in CI, and reports what needs changing:

- a floor that SPEC 0 has moved past (time to raise it);
- a released Python version that CI or the classifiers do not cover yet.

With --apply it also makes the changes: the floors, classifiers and ruff
target in pyproject.toml, the Python versions and the NumPy pin in the
workflows, and the requirements in docs/getting-started.md.  Every edit must
find what it replaces, so a file that has drifted from the expected layout
stops the update instead of being skipped.

Release dates come from endoflife.date (Python) and PyPI (packages).  Uses
the standard library only.

    python .github/scripts/spec0_check.py [--today YYYY-MM-DD] [--apply] [--report FILE]

With --report, the report is written to FILE when something is due, and
``due=true`` or ``due=false`` is appended to $GITHUB_OUTPUT if it is set.
"""

import argparse
import datetime as dt
import json
import os
import re
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOWS = ROOT / ".github" / "workflows"
GETTING_STARTED = ROOT / "docs" / "getting-started.md"
PYTHON_WINDOW_YEARS = 3
PACKAGE_WINDOW_YEARS = 2
UPCOMING_DAYS = 90
CLASSIFIER = "Programming Language :: Python :: "


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "felcu-spec0-check"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _version(text):
    return tuple(int(part) for part in text.split("."))


def _text(version):
    return ".".join(map(str, version))


def _add_years(date, years):
    # SPEC 0's own schedule counts a year as 365 days, so a window spanning
    # a leap day ends a day earlier than the anniversary.
    return date + dt.timedelta(days=365 * years)


# ---------------------------------------------------------------------------
# Release dates and SPEC 0 windows
# ---------------------------------------------------------------------------


def python_releases():
    """``{(3, 13): release date}`` for every Python 3 feature release."""
    releases = {}
    for row in _get_json("https://endoflife.date/api/python.json"):
        version = _version(row["cycle"])
        if version[0] == 3 and row.get("releaseDate"):
            releases[version] = dt.date.fromisoformat(row["releaseDate"])
    return releases


def package_releases(name):
    """``{(2, 2): release date}``: the first upload of each X.Y.0 release on PyPI."""
    releases = {}
    for text, files in _get_json(f"https://pypi.org/pypi/{name}/json")["releases"].items():
        match = re.fullmatch(r"(\d+)\.(\d+)\.0", text)
        if match and files:
            uploaded = min(dt.datetime.fromisoformat(f["upload_time"]).date() for f in files)
            releases[(int(match[1]), int(match[2]))] = uploaded
    return releases


def required_minimum(releases, years, today):
    """The oldest release still inside its window, and the date it drops out."""
    supported = {v: _add_years(d, years) for v, d in releases.items() if _add_years(d, years) > today}
    oldest = min(supported)
    return oldest, supported[oldest]


# ---------------------------------------------------------------------------
# Current versions
# ---------------------------------------------------------------------------


def _floor(requirement, name):
    match = re.search(rf"{re.escape(name)}\s*>=\s*(\d+\.\d+)", requirement)
    if match is None:
        raise SystemExit(f"No '{name}>=X.Y' floor in {requirement!r}.")
    return _version(match[1])


def _matrix(ci):
    match = re.search(r"python-version:\s*\[([^\]]*)\]", ci)
    if match is None:
        raise SystemExit("No python-version matrix in ci.yml.")
    return match


def current_versions():
    """Floors from pyproject.toml, and the Python versions in the classifiers and the CI matrix."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    floors = {
        "Python": _floor(project["requires-python"], ""),
        "numpy": _floor(" ".join(project["dependencies"]), "numpy"),
        "matplotlib": _floor(" ".join(project["optional-dependencies"]["plots"]), "matplotlib"),
    }
    classifiers = {
        _version(c[len(CLASSIFIER) :])
        for c in project["classifiers"]
        if c.startswith(CLASSIFIER) and re.fullmatch(r"3\.\d+", c[len(CLASSIFIER) :])
    }
    tested = {_version(v) for v in re.findall(r"\d+\.\d+", _matrix((WORKFLOWS / "ci.yml").read_text())[1])}
    return floors, classifiers, tested


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


def check(today):
    """What is due now, what is coming up, and the versions an update should set."""
    floors, classifiers, tested = current_versions()
    releases = {
        "Python": python_releases(),
        "numpy": package_releases("numpy"),
        "matplotlib": package_releases("matplotlib"),
    }

    due, upcoming, target = [], [], {}
    for name, floor in floors.items():
        years = PYTHON_WINDOW_YEARS if name == "Python" else PACKAGE_WINDOW_YEARS
        minimum, drops = required_minimum(releases[name], years, today)
        target[name] = max(floor, minimum)
        if floor < minimum:
            due.append(f"Raise the {name} floor from {_text(floor)} to {_text(minimum)}.")
        if (drops - today).days <= UPCOMING_DAYS:
            following = min((v for v in releases[name] if v > minimum), default=None)
            moves = f"to {_text(following)}" if following else "past the newest release"
            upcoming.append(f"{name} {_text(minimum)} drops out on {drops}: the floor moves {moves}.")

    newest = max(releases["Python"])
    target["newest Python"] = newest
    for where, versions in (("the CI test matrix", tested), ("the classifiers", classifiers)):
        if newest not in versions:
            due.append(f"Python {_text(newest)} is released but not in {where}.")
        stale = sorted(v for v in versions if v < target["Python"])
        if stale:
            due.append(f"Remove Python {', '.join(map(_text, stale))} from {where}: below the floor.")
    return due, upcoming, target


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _sub(pattern, replacement, text, where, count=0):
    new, n = re.subn(pattern, replacement, text, count=count, flags=re.MULTILINE)
    if n == 0:
        raise SystemExit(f"Could not find {pattern!r} in {where}; update it by hand.")
    return new


def _pythons(lo, hi):
    return [(3, minor) for minor in range(lo[1], hi[1] + 1)]


def apply(target):
    """Write the *target* versions into every file that states them."""
    python, numpy, matplotlib, newest = (target[k] for k in ("Python", "numpy", "matplotlib", "newest Python"))
    supported = _pythons(python, newest)

    text = PYPROJECT.read_text()
    text = _sub(r'requires-python = ">=[\d.]+"', f'requires-python = ">={_text(python)}"', text, PYPROJECT)
    classifiers = "".join(f'    "{CLASSIFIER}{_text(v)}",\n' for v in supported)
    text = _sub(rf'(?:    "{CLASSIFIER}3\.\d+",\n)+', classifiers, text, PYPROJECT, count=1)
    text = _sub(r'"numpy>=[\d.]+"', f'"numpy>={_text(numpy)}"', text, PYPROJECT)
    text = _sub(r'"matplotlib>=[\d.]+"', f'"matplotlib>={_text(matplotlib)}"', text, PYPROJECT)
    text = _sub(r'target-version = "py\d+"', f'target-version = "py3{python[1]}"', text, PYPROJECT)
    PYPROJECT.write_text(text)

    ci_path = WORKFLOWS / "ci.yml"
    ci = ci_path.read_text()
    # Keep any pre-release already in the matrix, beyond the newest release.
    prereleases = sorted(v for v in (_version(s) for s in re.findall(r"\d+\.\d+", _matrix(ci)[1])) if v > newest)
    matrix = ", ".join(f'"{_text(v)}"' for v in supported + prereleases)
    ci = _sub(r"python-version:\s*\[[^\]]*\]", f"python-version: [{matrix}]", ci, ci_path)
    ci = _sub(r'"numpy==[\d.]+\.\*"', f'"numpy=={_text(numpy)}.*"', ci, ci_path)
    ci_path.write_text(ci)

    # Every job that pins one Python uses the oldest supported version.
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text()
        text = re.sub(r'python-version: "3\.\d+"', f'python-version: "{_text(python)}"', text)
        text = re.sub(r"name: Set up Python 3\.\d+$", f"name: Set up Python {_text(python)}", text, flags=re.M)
        path.write_text(text)

    text = GETTING_STARTED.read_text()
    text = _sub(
        r"needs Python [\d.]+\+ and NumPy [\d.]+\+",
        f"needs Python {_text(python)}+ and NumPy {_text(numpy)}+",
        text,
        GETTING_STARTED,
    )
    GETTING_STARTED.write_text(text)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(due, upcoming, today, applied):
    lines = [f"SPEC 0 check on {today}.", ""]
    if due:
        lines += ["**Due now:**", ""] + [f"- {item}" for item in due] + [""]
    else:
        lines += ["Nothing is due.", ""]
    if upcoming:
        lines += [f"**In the next {UPCOMING_DAYS} days:**", ""] + [f"- {item}" for item in upcoming] + [""]
    if due and applied:
        lines += [
            "This updates `pyproject.toml`, the workflows and `docs/getting-started.md` to match.  A higher "
            "ruff `target-version` or a new Python may also need code or dependency changes: check CI.",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--apply", action="store_true", help="update the files when something is due")
    parser.add_argument("--report", type=Path, help="write the report here when something is due")
    args = parser.parse_args()

    due, upcoming, target = check(args.today)
    if due and args.apply:
        apply(target)
    text = report(due, upcoming, args.today, args.apply)
    print(text)
    if args.report is not None and due:
        args.report.write_text(text)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"due={'true' if due else 'false'}\n")
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(text)


if __name__ == "__main__":
    main()
