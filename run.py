"""
Send a run from the command line: the base run of config/presets.json with presets laid over it.

    python run.py                                   lists the base and the presets
    python run.py -d sod -m nano                    a data preset and a model preset
    python run.py -m nano                           the base's data, that model
    python run.py -d citizen                        that data, the base's model
    python run.py -d sod -m nano --sample 200       200 subjects of it; a trial
    python run.py -d sod -m nano --question neutral with a field replaced
    python run.py -d sod -m nano --show             the system prompt and one prompt; no call
    python run.py -d sod -m nano --one              one call, printed; nothing saved
    python run.py -d sod -m nano --yes              send without asking
    python run.py -d sod -m nano --check            look over the saved run; no call

Later replaces earlier: base, `-d`, `-m`, single fields. Every field of a run is an option;
`none` stands for no value (`--temperature none`). Anything but the Fake client asks before
sending. Writes under `results/<topic>/`, then prints a check of what was saved.
"""
import argparse

from dotenv import load_dotenv

from panel import paths
from panel.run import ResultStore, execute, resolve
from panel.run.check import check, means, report
from panel.run.fake import FakeClient
from panel.run.presets import DATA, FIELDS, MODEL, build_spec, load_presets, parse

PRESETS = paths.CONFIG_DIR / "presets.json"


def listing(base: dict, presets: dict, kind: tuple) -> str:
    rows = {"(base)": {}, **presets}
    width = max(map(len, rows))
    return "\n".join(f"  {name:<{width}}  " + "  ".join(f"{field}={ {**base, **preset}[field] }" for field in kind) for name, preset in rows.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a run: the base run with a data preset, a model preset and single fields laid over it.", allow_abbrev=False)
    parser.add_argument("-d", "--data-preset", dest="data", help="a data preset in config/presets.json")
    parser.add_argument("-m", "--model-preset", dest="model_preset", help="a model preset in config/presets.json")
    for field in FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", dest=field)
    parser.add_argument("--show", action="store_true", help="print the system prompt and one prompt; no call")
    parser.add_argument("--one", action="store_true", help="make one call and print it; nothing saved")
    parser.add_argument("--yes", action="store_true", help="send without asking")
    parser.add_argument("--check", action="store_true", help="look over the saved run; no call")
    args = parser.parse_args()

    changes = {field: parse(field, getattr(args, field)) for field in FIELDS if getattr(args, field) is not None}
    presets = load_presets(PRESETS)
    if args.data is None and args.model_preset is None and not changes:
        print(f"data (-d)\n{listing(presets.base, presets.data, DATA)}\n\nmodels (-m)\n{listing(presets.base, presets.models, MODEL)}")
        return
    if args.data is not None and args.data not in presets.data:
        parser.error(f"no data preset '{args.data}' in {PRESETS}; it has {list(presets.data)}")
    if args.model_preset is not None and args.model_preset not in presets.models:
        parser.error(f"no model preset '{args.model_preset}' in {PRESETS}; it has {list(presets.models)}")

    spec = build_spec(presets.base, presets.data.get(args.data), presets.models.get(args.model_preset), changes)
    run = resolve(spec)
    store = ResultStore(paths.RESULTS_DIR, spec.topic)
    if args.check:
        print(f"{report(check(run, store), len(run.topic.options))}\n\n{means(run.topic.rename(store.frame(run.stem)), run.topic).round(3).to_string()}")
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
    print(f"{report(check(run, store), len(run.topic.options))}\n\n{means(run.topic.rename(store.frame(run.stem)), run.topic).round(3).to_string()}")


if __name__ == "__main__":
    main()
