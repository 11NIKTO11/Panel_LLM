"""
Prepare a dataset from the command line.

    python prepare.py                               asks for everything
    python prepare.py sod                           values and every description
    python prepare.py sod -d default                values and one description
    python prepare.py sod -d default --prompts-only keep the stored values, render again

Naming the dataset on the command line asks nothing. Writes under `data/<dataset>/`, and says
for each prompt table how many prompts are identical to another subject's.
"""
import argparse

from panel.dataset import dataset_names, load_dataset
from panel.prepare import VALUES, count_prompts, prepare


def pick(question: str, options: list, several: bool = False) -> list:
    """Ask for one option, or several by number or name; Enter takes all when `several`."""
    for number, option in enumerate(options, 1):
        print(f"  {number}. {option}")
    while True:
        answer = input(f"{question}{' [all]' if several else ''}: ").replace(",", " ").split()
        if not answer and several:
            return options
        chosen = [options[int(item) - 1] if item.isdigit() and 0 < int(item) <= len(options) else item for item in answer]
        if chosen and all(item in options for item in chosen) and (several or len(chosen) == 1):
            return chosen
        print("  not one of the options")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a dataset: values table and prompt tables.")
    parser.add_argument("dataset", nargs="?", help="a folder under datasets/; asked for when omitted")
    parser.add_argument("-d", "--descriptions", nargs="+", help="descriptions to render; all when omitted")
    parser.add_argument("--prompts-only", action="store_true", help="render over the stored values")
    args = parser.parse_args()

    name, descriptions, rebuild = args.dataset, args.descriptions, not args.prompts_only
    if name is None:
        name = pick("dataset", dataset_names())[0]
        descriptions = pick("descriptions", list(load_dataset(name).descriptions), several=True)
        rebuild = input("rebuild the values table from the raw source? [Y/n]: ").strip().lower() not in ("n", "no")

    for what, path in prepare(name, descriptions, rebuild).items():
        print(f"{what} -> {path}")
        if what != VALUES:
            count = count_prompts(name, what)
            print(f"  {count.prompts} prompts, {count.identical} identical to another subject's, {count.distinct} distinct")


if __name__ == "__main__":
    main()
