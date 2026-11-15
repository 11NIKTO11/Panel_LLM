"""
Pack a saved run from the command line.

    python pack.py                                  lists every topic's saved runs
    python pack.py election_2021 haiku-5-5          the run whose stem contains that text
    python pack.py election_2021 haiku-5-5 --zip    and a zip of the pack beside it
    python pack.py election_2021 haiku-5-5 --copy   the saved analysis/ and plots/ as they are, none made again
    python pack.py election_2021 haiku-5-5 --to D:/packs
    python pack.py --verify packs/election_2021/<stem>

Writes `packs/<topic>/<stem>/`: the run, the dataset and topic files it names, its tables and
figures made again, and `MANIFEST.json` with the sha256 of every file (`panel/pack.py`). A pack
of a survey run holds the survey's values - mind where it goes.
"""
import argparse
from pathlib import Path

from cli import list_runs
from panel import paths
from panel.analysis import find_stem
from panel.pack import archive, pack, verify
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a saved run with everything it was built from.")
    parser.add_argument("topic", nargs="?", help="a folder under results/")
    parser.add_argument("run", nargs="?", help="a stem, or any text only one stem contains")
    parser.add_argument("--to", type=Path, default=paths.PACKS_DIR, help="the folder of packs; packs/ by default")
    parser.add_argument("--copy", action="store_true", help="copy the saved tables and figures, make none again")
    parser.add_argument("--zip", action="store_true", help="also write <stem>.zip beside the pack")
    parser.add_argument("--verify", type=Path, metavar="PACK", help="check a pack's files against its manifest instead")
    args = parser.parse_args()

    if args.verify:
        wrong = verify(args.verify)
        print("\n".join(f"  {name}" for name in wrong) if wrong else "every file is what the manifest says")
        raise SystemExit(1 if wrong else 0)
    if args.run is None:
        list_runs(args.topic)
        return

    store = ResultStore(paths.RESULTS_DIR, args.topic)
    folder = pack(store, find_stem(store, args.run), args.to, derive=not args.copy)
    print(f"-> {folder}")
    if args.zip:
        print(f"-> {archive(folder)}")


if __name__ == "__main__":
    main()
