"""Check the supported Python and dependency versions against SPEC 0.

SPEC 0 (https://scientific-python.org/specs/spec-0000/) supports each Python
feature release for 3 years and each release of a core package for 2 years.
This compares those windows with the floors in pyproject.toml and the Python
versions tested in CI, and reports what needs changing:

- a floor that SPEC 0 has moved past (time to raise it);
- a released Python version that CI or the classifiers do not cover yet.

Release dates come from endoflife.date (Python) and PyPI (packages).  Uses
the standard library only.

    python .github/scripts/spec0_check.py [--today YYYY-MM-DD] [--report FILE]

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
PYTHON_WINDOW_YEARS = 3
PACKAGE_WINDOW_YEARS = 2
UPCOMING_DAYS = 90


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


def _floor(requirement, name):
    match = re.search(rf"{re.escape(name)}\s*>=\s*(\d+\.\d+)", requirement)
    if match is None:
        raise SystemExit(f"No '{name}>=X.Y' floor in {requirement!r}.")
    return _version(match[1])


def current_versions():
    """Floors from pyproject.toml, and the Python versions in the classifiers and the CI matrix."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    floors = {
        "Python": _floor(project["requires-python"], ""),
        "numpy": _floor(" ".join(project["dependencies"]), "numpy"),
        "matplotlib": _floor(" ".join(project["optional-dependencies"]["plots"]), "matplotlib"),
    }
    prefix = "Programming Language :: Python :: "
    classifiers = {_version(c[len(prefix) :]) for c in project["classifiers"] if re.fullmatch(r".*3\.\d+", c)}
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    matrix = re.search(r"python-version:\s*\[([^\]]*)\]", ci)
    if matrix is None:
        raise SystemExit("No python-version matrix in ci.yml.")
    tested = {_version(v) for v in re.findall(r"\d+\.\d+", matrix[1])}
    return floors, classifiers, tested


def check(today):
    floors, classifiers, tested = current_versions()
    releases = {"Python": python_releases(), "numpy": package_releases("numpy")}
    releases["matplotlib"] = package_releases("matplotlib")

    due, upcoming = [], []
    for name, floor in floors.items():
        years = PYTHON_WINDOW_YEARS if name == "Python" else PACKAGE_WINDOW_YEARS
        minimum, drops = required_minimum(releases[name], years, today)
        if floor < minimum:
            due.append(f"Raise the {name} floor from {_text(floor)} to {_text(minimum)}.")
        if (drops - today).days <= UPCOMING_DAYS:
            following = min((v for v in releases[name] if v > minimum), default=None)
            target = f"to {_text(following)}" if following else "past the newest release"
            upcoming.append(f"{name} {_text(minimum)} drops out on {drops}: the floor moves {target}.")

    newest = max(releases["Python"])
    for where, versions in (("the CI test matrix", tested), ("the classifiers", classifiers)):
        if newest not in versions:
            due.append(f"Python {_text(newest)} is released but not in {where}.")
        stale = sorted(v for v in versions if v < floors["Python"])
        if stale:
            due.append(f"Remove Python {', '.join(map(_text, stale))} from {where}: below the floor.")
    return due, upcoming


def report(due, upcoming, today):
    lines = [f"SPEC 0 check on {today}.", ""]
    if due:
        lines += ["**Due now:**", ""] + [f"- {item}" for item in due] + [""]
    else:
        lines += ["Nothing is due.", ""]
    if upcoming:
        lines += [f"**In the next {UPCOMING_DAYS} days:**", ""] + [f"- {item}" for item in upcoming] + [""]
    lines.append("Every place a raise touches is listed under *Supported versions* in `docs/development.md`.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--report", type=Path, help="write the report here when something is due")
    args = parser.parse_args()

    due, upcoming = check(args.today)
    text = report(due, upcoming, args.today)
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
