"""Download a bounded, complete-message prefix of Nasdaq's public 2019 sample."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

from models.lob.itch import messages

URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/01302019.NASDAQ_ITCH50.gz"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/real-data/nasdaq/prefix.itch"))
    parser.add_argument("--messages", type=int, default=500000)
    args = parser.parse_args()
    if not 1 <= args.messages <= 5000000:
        parser.error("--messages must be in 1..5000000; this is a bounded smoke-test downloader")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "prefix.itch"
        count = 0
        with urllib.request.urlopen(URL, timeout=30) as response, gzip.GzipFile(fileobj=response) as stream:
            with staged.open("wb") as target:
                for body in messages(stream):
                    target.write(len(body).to_bytes(2, "big") + body)
                    count += 1
                    if count == args.messages:
                        break
        if count != args.messages:
            raise ValueError("Source ended before requested message count")
        metadata = {"schema_version": 1, "url": URL, "framed_messages": count,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "selection": "first N complete framed messages from start of archive",
                    "complete_archive": False, "gzip_trailer_verified": False,
                    "sha256": sha256_file(staged), "bytes": staged.stat().st_size}
        staged.replace(args.output)
        args.output.with_suffix(".download.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
