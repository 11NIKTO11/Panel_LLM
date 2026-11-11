"""
Send a run from the command line.

    python run.py                                   lists the named runs
    python run.py fake                              a named run from config/runs.json
    python run.py nano --question neutral           a named run with fields replaced
    python run.py --dataset sod --description default --topic election_2021 --framing cs \\
                  --question default --client Fake --model fake-model
                                                    no name: every field from the arguments
    python run.py nano --show                       the system prompt and one prompt; no call
    python run.py nano --one                        one call, printed; nothing saved
    python run.py nano --yes                        send without asking
    python run.py nano --check                      look over the saved run; no call

Every field of a run is an option; `none` stands for no value (`--temperature none`).
Anything but the Fake client asks before sending. Writes under `results/<topic>/`, then prints
a check of what was saved.
"""
import argparse

from dotenv import load_dotenv

from panel import paths
from panel.run import ResultStore, execute, resolve
from panel.run.check import check, report
from panel.run.fake import FakeClient
from panel.run.named import FIELDS, build_spec, load_runs, parse

RUNS = paths.CONFIG_DIR / "runs.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a run: a named one, one from arguments, or a named one changed.")
    parser.add_argument("name", nargs="?", help="a run in config/runs.json")
    for field in FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", dest=field)
    parser.add_argument("--show", action="store_true", help="print the system prompt and one prompt; no call")
    parser.add_argument("--one", action="store_true", help="make one call and print it; nothing saved")
    parser.add_argument("--yes", action="store_true", help="send without asking")
    parser.add_argument("--check", action="store_true", help="look over the saved run; no call")
    args = parser.parse_args()

    changes = {field: parse(field, getattr(args, field)) for field in FIELDS if getattr(args, field) is not None}
    runs = load_runs(RUNS)
    if args.name is None and not changes:
        for name, named in runs.items():
            print(f"  {name:<12} {resolve(build_spec(named, {})).stem}")
        return
    if args.name is not None and args.name not in runs:
        parser.error(f"no run '{args.name}' in {RUNS}; it has {list(runs)}")

    spec = build_spec(runs[args.name] if args.name else None, changes)
    run = resolve(spec)
    store = ResultStore(paths.RESULTS_DIR, spec.topic)
    if args.check:
        print(report(check(run, store), len(run.topic.options)))
        return
    saved = store.paths(run.stem).record.exists()
    print(f"{run.stem}: {len(run.calls)} calls over {len(run.rows)} subjects{' - replaces a saved run' if saved else ''}")

    subject = run.calls.iloc[0]
    if args.show:
        print(f"\n{run.framing.instructions}\n\n{run.prompt(subject)}")
        return
    load_dotenv()
    client = run.client_type(**({"effort": spec.effort} if spec.effort else {}))
    if args.one:
        print(client.call_llm(spec.model, run.prompt(subject), run.response_model, spec.temperature, run.framing.instructions))
        return
    if not (args.yes or isinstance(client, FakeClient)):
        if input(f"send {len(run.calls)} calls to {spec.client} {spec.model}? [y/N] ").strip().lower() != "y":
            return
    written = execute(run, store, client)
    print(f"-> {written.results_csv}\n")
    print(report(check(run, store), len(run.topic.options)))


if __name__ == "__main__":
    main()
