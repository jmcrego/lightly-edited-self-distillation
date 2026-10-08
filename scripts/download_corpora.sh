#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 SOURCE TARGET [OUTPUT_DIRECTORY]"
    echo "Example: $0 en fr"
    echo "Install dependencies: python3 -m pip install opustools"
    echo "Optional: PYTHON=/path/to/python (must have opustools installed)"
    echo "Optional: OPUS_FORMATS=moses for sentence pairs only (default: xml,moses)"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 || $# -gt 3 || $1 == "$2" ]]; then
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir=${3:-"$script_dir/../corpora"}
python_bin=${PYTHON:-python3}

# Resolve and pin releases through the same API used by opus_get. Keep XML
# document/alignment archives as well as the convenient Moses sentence pairs.
"$python_bin" - "$1" "$2" "$output_dir" <<'PY'
import hashlib
import gzip
import importlib.metadata
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from opustools.opus_get import OpusGet
except ImportError:
    sys.exit("Install OPUS Tools first: python3 -m pip install opustools")

source, target, output = sys.argv[1:]
formats = os.environ.get("OPUS_FORMATS", "xml,moses").split(",")
if not formats or any(value not in ("xml", "moses") for value in formats):
    sys.exit("OPUS_FORMATS must be xml, moses, or xml,moses")
root = Path(output).resolve()
pair = "-".join(sorted((source, target)))
for corpus in ("JRC-Acquis", "EMEA", "KDE4", "ECB"):
    directory = root / corpus / pair
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["corpus"], manifest["source"], manifest["target"]) != (
            corpus, source, target
        ):
            sys.exit(f"Manifest language direction mismatch: {manifest_path}")
        release = manifest["release"]
    else:
        query = OpusGet(source=source, target=target, directory=corpus,
                        preprocess="moses", download_dir=str(directory))
        resources, _ = query.get_corpora_data()
        versions = {resource["version"] for resource in resources}
        if len(versions) != 1:
            sys.exit(f"Expected one release for {corpus}/{pair}, found {versions}")
        release = versions.pop()
        manifest = {
            "corpus": corpus, "source": source, "target": target,
            "release": release,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "opustools_version": importlib.metadata.version("opustools"),
            "resources": [], "complete": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Downloading {corpus} {release}: {source} -> {target}", flush=True)
    recorded = []
    for preprocessing in formats:
        query = OpusGet(source=source, target=target, directory=corpus,
                        release=release, preprocess=preprocessing,
                        download_dir=str(directory), suppress_prompts=True)
        resources = query.get_response(query.url)["corpora"]
        if not resources:
            sys.exit(f"No {preprocessing} resources for {corpus}/{release}/{pair}")
        missing = [resource for resource in resources
                   if not Path(query.make_file_name(resource)).exists()]
        if missing:
            # OPUS Tools otherwise flushes terminal output for every 8 KiB block.
            original_progress = query.progress_status
            last_progress = [0.0]
            def progress(count, block_size, total_size):
                now = time.monotonic()
                if now - last_progress[0] >= 1 or count * block_size >= total_size:
                    original_progress(count, block_size, total_size)
                    last_progress[0] = now
            query.progress_status = progress
            query.download(missing, "selected resources")
        print(f"Validating {corpus} {preprocessing} archives...", flush=True)
        for resource in resources:
            path = Path(query.make_file_name(resource))
            if not path.is_file() or path.stat().st_size == 0:
                sys.exit(f"Missing or empty download: {path}")
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    bad_member = archive.testzip()
                    if bad_member:
                        sys.exit(f"Corrupt archive member: {path}: {bad_member}")
            elif path.suffix == ".gz":
                with gzip.open(path, "rb") as handle:
                    while handle.read(1024 * 1024):
                        pass
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            recorded.append({"preprocessing": preprocessing,
                             "url": resource["url"], "file": path.name,
                             "bytes": path.stat().st_size,
                             "sha256": digest.hexdigest()})
    manifest["resources"] = recorded
    manifest["formats"] = formats
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Recorded release and checksums: {manifest_path}", flush=True)
PY
