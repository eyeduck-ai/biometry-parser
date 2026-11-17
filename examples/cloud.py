"""Explicit opt-in cloud fallback for a caller-supplied, prepared input.

Set GEMINI_API_KEY or PADDLEOCR_ACCESS_TOKEN in the process environment.
This example performs no automatic deidentification.
"""
from argparse import ArgumentParser
from pathlib import Path
from biometry_parser import BiometryParser


def main():
    args = ArgumentParser(description=__doc__)
    args.add_argument("source")
    args.add_argument("--backend", choices=["gemini", "paddle_api"], default="gemini")
    args.add_argument("--model")
    args.add_argument("--output", default="result.json")
    args.add_argument("--allow-cloud", action="store_true", required=True,
                      help="Explicitly authorize sending image fallback pages to the selected service")
    options = args.parse_args()
    parser = BiometryParser(backend=options.backend, model=options.model, allow_cloud=options.allow_cloud)
    result = parser.parse(options.source)
    Path(options.output).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
