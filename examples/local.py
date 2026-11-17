"""Parse a caller-supplied report using native PDF first, then local Small."""
from argparse import ArgumentParser
from pathlib import Path
from biometry_parser import BiometryParser


def main():
    args = ArgumentParser(description=__doc__)
    args.add_argument("source")
    args.add_argument("--output", default="result.json")
    args.add_argument("--model-cache")
    args.add_argument("--page-strategy", choices=["all", "auto", "biometry_first"], default="all")
    options = args.parse_args()
    parser = BiometryParser(model_cache=options.model_cache, threads=2)
    result = parser.parse(options.source, page_strategy=options.page_strategy)
    Path(options.output).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
