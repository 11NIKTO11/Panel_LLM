"""
Write the runs saved in the legacy shape again as runs of today.

    python migrate_legacy.py                     lists what would be migrated, writes nothing
    python migrate_legacy.py --write             writes the migrated runs beside the legacy ones
    python migrate_legacy.py --write --to <dir>  writes them under another results folder, for a trial

The legacy files are only read. A migrated run has a new stem, so nothing is overwritten;
once the migrated runs are checked, the legacy files are the ones to remove. With `--write`
the dataset's stored description, `data/sod/prompts/legacy.csv`, is written first if it is
not there. See `panel/run/legacy.py`.
"""
import argparse

from panel import paths
from panel.run import ResultStore, resolve
from panel.run.legacy import Legacy, legacy_stems, migrate, spec_of, write_stored_prompts
from panel.topic import load_question, load_topic

TOPIC = "election_2021_legacy"
LEGACY = Legacy(dataset="sod", description="legacy", framing="cs", question="default")
PROMPTS = paths.DATA_DIR / "sod_legacy" / "prompts.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate the legacy runs to today's shape.")
    parser.add_argument("--write", action="store_true", help="write the migrated runs; without it only list them")
    parser.add_argument("--to", help="another results folder to write under; results/ by default")
    args = parser.parse_args()

    source = ResultStore(paths.RESULTS_DIR, TOPIC)
    stems = legacy_stems(source)
    if not args.write:
        for stem in stems:
            spec = spec_of(source.record(stem), LEGACY)
            print(f"  {stem}\n    -> {spec.shape}, {spec.client} {spec.model}, sample {spec.sample}")
        print(f"{len(stems)} legacy runs in {source.topic_dir}; --write migrates them")
        return

    question = load_question(load_topic(TOPIC), "cs", LEGACY.question).text
    print(f"stored description -> {write_stored_prompts(PROMPTS, LEGACY, question)}")
    target = ResultStore(args.to or paths.RESULTS_DIR, TOPIC)
    for stem in stems:
        done = migrate(source, stem, target, LEGACY)
        normalized = "no legacy normalized table" if done.normalized_differs is None else f"{done.normalized_differs} normalized cells differ"
        print(f"  {done.source}\n    -> {done.stem}: {done.answers} answers, {done.prompts_checked} saved prompts checked, {normalized}")
    print(f"{len(stems)} runs migrated into {target.topic_dir}")


if __name__ == "__main__":
    main()
