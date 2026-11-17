"""Small command-line wrapper around the reusable parser API."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .config import load_config
from .models import ExtractionResult


def _write(value, output=None):
    payload=value.model_dump_json(indent=2) if hasattr(value,"model_dump_json") else json.dumps(value,ensure_ascii=False,indent=2)
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(payload+"\n",encoding="utf-8")
    else:
        print(payload)


def _parser():
    parser=argparse.ArgumentParser(prog="biometry-parser",description="Structured IOLMaster 700 biometry and IOL calculation extraction")
    parser.add_argument("--version",action="version",version="biometry-parser 0.1.0")
    commands=parser.add_subparsers(dest="command",required=True)
    extract=commands.add_parser("extract")
    extract.add_argument("source")
    extract.add_argument("-c","--config")
    select=extract.add_mutually_exclusive_group()
    select.add_argument("-w","--workflow")
    select.add_argument("--backend",choices=("local","native","gemini","gemini_flash","paddle_api"))
    extract.add_argument("-o","--output")
    extract.add_argument("--pages",nargs="+",type=int,help="Original one-based pages")
    extract.add_argument("--page-strategy",choices=("all","auto","biometry_first"),default="all")
    extract.add_argument("--force-ocr",action="store_true",help="Disable native-text shortcut for the selected workflow")
    extract.add_argument("--allow-cloud",action="store_true",help="Explicitly allow the selected remote recognizer to receive images")
    models=commands.add_parser("models")
    model_actions=models.add_subparsers(dest="action",required=True)
    model_actions.add_parser("list")
    download=model_actions.add_parser("download");download.add_argument("model_id");download.add_argument("--cache-dir")
    providers=commands.add_parser("providers");providers.add_subparsers(dest="action",required=True).add_parser("list")
    doctor=commands.add_parser("doctor");doctor.add_argument("-c","--config");doctor.add_argument("--profile",action="append")
    schema=commands.add_parser("schema");schema.add_argument("-o","--output")
    return parser


def main(argv=None) -> int:
    args=_parser().parse_args(argv)
    try:
        if args.command=="extract":
            from .client import extract
            mapping={"local":"local","native":"native_only","gemini":"gemini","gemini_flash":"gemini_flash","paddle_api":"paddle_api"}
            workflow=args.workflow or mapping.get(args.backend)
            config=load_config(args.config,workflow=workflow)
            if args.force_ocr:
                config=config.model_copy(deep=True)
                config.workflows[config.workflow].native_first=False
            result=extract(args.source,config,pages=args.pages,page_strategy=args.page_strategy,allow_cloud=args.allow_cloud)
            _write(result,args.output)
            # Producing candidates does not mean they have clinical verification.
            # Verification remains explicit in the JSON, not encoded as shell success.
            if not result.runs or any(run.outcome != "ok" for run in result.runs):
                return 2
            return 0 if any(run.observations or run.iol_calculations for run in result.runs) else 2
        if args.command=="models":
            from .model_store import download_model,load_manifest
            _write(load_manifest() if args.action=="list" else download_model(args.model_id,args.cache_dir))
        elif args.command=="providers":
            from .providers import list_providers
            _write(list_providers())
        elif args.command=="schema":
            _write(ExtractionResult.model_json_schema(mode="serialization"),args.output)
        elif args.command=="doctor":
            from .worker_client import WorkerClient
            config=load_config(args.config)
            profiles=args.profile or config.workflows[config.workflow].profiles
            checks=[]
            for name in profiles:
                if name not in config.profiles:
                    raise ValueError("unknown_profile")
                client=WorkerClient(config.profiles[name],profile_id=name)
                try:
                    checks.append({"profile":name,"available":True,"descriptor":client.start()})
                except Exception as exc:
                    checks.append({"profile":name,"available":False,"code":getattr(exc,"code","provider_error")})
                finally:
                    client.close()
            _write(checks)
            return 0 if all(check["available"] for check in checks) else 1
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        code=getattr(exc,"code","invalid_input" if isinstance(exc,(ValueError,FileNotFoundError)) else "execution_failed")
        print(f"biometry-parser: {code}; check configuration, input and optional dependencies.",file=sys.stderr)
        return 1
