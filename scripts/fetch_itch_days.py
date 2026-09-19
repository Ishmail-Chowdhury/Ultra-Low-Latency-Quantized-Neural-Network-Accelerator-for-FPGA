"""Stream complete public Nasdaq days, retaining one symbol and system events.

The entire gzip stream is consumed and its CRC checked. SHA-256 covers all
downloaded compressed bytes; the retained file has its own SHA-256. No credentials.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time
import urllib.request
import subprocess
import shutil

from models.lob.itch import LENGTHS
from scripts.fetch_itch_sample import sha256_file


DAYS = {
    "2019-07-30": "07302019.NASDAQ_ITCH50.gz",
    "2019-08-30": "08302019.NASDAQ_ITCH50.gz",
    "2019-10-18": "S101819-v50.txt.gz",
    "2019-12-30": "12302019.NASDAQ_ITCH50.gz",
}
BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"


def download_ranges(url, destination):
    """Use ordinary HTTP byte ranges, with four connections and resumable parts."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
        total = int(response.headers["Content-Length"])
        identity = {"url": url, "bytes": total, "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified")}
    part_dir = destination.with_suffix(".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    manifest = part_dir / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Archive changed; refusing to reuse previous byte ranges")
    manifest.write_text(json.dumps(identity))
    chunk_size = 64 * 1024 * 1024

    def part(start):
        stop = min(start + chunk_size, total) - 1
        path = part_dir / str(start)
        if path.exists() and path.stat().st_size == stop - start + 1:
            return path
        headers = {"Range": f"bytes={start}-{stop}"}
        if identity["etag"]:
            headers["If-Match"] = identity["etag"]
        request = urllib.request.Request(url, headers=headers)
        staged = path.with_suffix(".tmp")
        with urllib.request.urlopen(request, timeout=120) as response, staged.open("wb") as target:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{total}":
                raise ValueError("Server did not honor requested archive byte range")
            shutil.copyfileobj(response, target, 1024 * 1024)
        if staged.stat().st_size != stop - start + 1:
            raise ValueError("Truncated archive range")
        staged.replace(path)
        return path

    with ThreadPoolExecutor(4) as pool:
        paths = []
        for path in pool.map(part, range(0, total, chunk_size)):
            paths.append(path)
            if len(paths) % 8 == 0:
                print(f"{destination.stem}: downloaded {min(len(paths)*chunk_size,total)/1e9:.2f}/{total/1e9:.2f} GB", flush=True)
    staged = destination.with_suffix(".tmp")
    with staged.open("wb") as target:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target, 1024 * 1024)
    staged.replace(destination)
    shutil.rmtree(part_dir)
    return total


class MeteredReader:
    def __init__(self, stream, day, total):
        self.stream, self.day, self.total = stream, day, total
        self.digest = hashlib.sha256()
        self.count = 0
        self.last_log = time.monotonic()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        if time.monotonic() - self.last_log > 30:
            print(f"{self.day}: {self.count / 1e9:.2f}/{self.total / 1e9:.2f} GB", flush=True)
            self.last_log = time.monotonic()
        return data


def filter_symbol(stream, target, symbol):
    """Chunked BinaryFILE scan; preserve source order and validate every frame."""
    locate = None
    pending = b""
    messages = retained = 0
    events = []
    terminated = False
    symbol_bytes = symbol.encode("ascii").ljust(8)
    while chunk := stream.read(1024 * 1024):
        data = pending + chunk
        offset = 0
        while offset + 2 <= len(data):
            if terminated:
                raise ValueError("Data after BinaryFILE session terminator")
            length = (data[offset] << 8) | data[offset + 1]
            if length == 0:
                terminated = True
                target.write(b"\x00\x00")
                offset += 2
                if offset != len(data):
                    raise ValueError("Data after BinaryFILE session terminator")
                break
            if not 11 <= length <= 100:
                raise ValueError("Invalid ITCH frame length")
            end = offset + 2 + length
            if end > len(data):
                break
            start = offset + 2
            kind = chr(data[start])
            if LENGTHS.get(kind) != length:
                raise ValueError(f"Unknown ITCH message or length {kind}/{length}")
            message_locate = data[start + 1:start + 3]
            if kind == "R" and data[start + 11:start + 19] == symbol_bytes:
                if locate is not None:
                    raise ValueError("Duplicate stock directory")
                locate = message_locate
            if kind == "S":
                events.append(chr(data[start + 11]))
            if messages == 0 and (kind != "S" or events != ["O"]):
                raise ValueError("Archive must start at Start of Messages")
            if kind == "S" or (locate is not None and message_locate == locate):
                target.write(data[offset:end])
                retained += 1
            messages += 1
            offset = end
        pending = data[offset:]
    if pending:
        raise ValueError("Truncated final ITCH message")
    if locate is None or events != ["O", "S", "Q", "M", "E", "C"]:
        raise ValueError(f"Missing symbol or incomplete daily system events: {events}")
    return {"all_messages": messages, "retained_messages": retained, "system_events": events,
            "binaryfile_terminator_seen": terminated}


def native_filter(stream, target, symbol, scanner):
    with subprocess.Popen([str(scanner.resolve()), symbol], stdin=subprocess.PIPE,
                          stdout=target, stderr=subprocess.PIPE) as process:
        try:
            while chunk := stream.read(1024 * 1024):
                process.stdin.write(chunk)
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait() != 0:
                raise ValueError(error)
            return json.loads(error)
        except BaseException:
            process.kill()
            process.wait()
            raise


def fetch(day, output, symbol="AAPL", scanner=None, ranged=False):
    url = BASE + DAYS[day]
    destination = output / f"{day}-{symbol}.itch"
    metadata_path = destination.with_suffix(".download.json")
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("complete_archive") and metadata.get("symbol") == symbol and
                metadata.get("url") == url and metadata.get("sha256") == sha256_file(destination)):
            print(f"{day}: reusing verified retained day", flush=True)
            return metadata
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staged = Path(temporary) / "symbol.itch"
        archive = output / f"{day}.gz"
        if ranged:
            length = download_ranges(url, archive)
            response = archive.open("rb")
        else:
            response = urllib.request.urlopen(url, timeout=120)
            length = int(response.headers["Content-Length"])
        with response:
            metered = MeteredReader(response, day, length)
            with gzip.GzipFile(fileobj=metered) as stream, staged.open("wb") as target:
                counts = (native_filter(stream, target, symbol, scanner) if scanner
                          else filter_symbol(stream, target, symbol))
            if metered.count != length:
                raise ValueError("Archive byte count differs from HTTP Content-Length")
        metadata = {"schema_version": 1, "url": url, "symbol": symbol, "date": day,
                    "complete_archive": True, "gzip_trailer_verified": True,
                    "compressed_bytes": metered.count, "compressed_sha256": metered.digest.hexdigest(),
                    "sha256": sha256_file(staged), "retained_bytes": staged.stat().st_size,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), **counts}
        staged.replace(destination)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if ranged:
            archive.unlink()  # Retained AAPL file and archive hash are the durable artifacts.
    print(f"{day}: complete; retained {counts['retained_messages']} messages", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/nasdaq-days/raw"))
    parser.add_argument("--day", choices=list(DAYS), action="append")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--scanner", type=Path, help="Optional compiled scripts/itch_filter.c executable")
    parser.add_argument("--ranged-download", action="store_true", help="Stage full gzip with resumable HTTP byte ranges")
    args = parser.parse_args()
    if not 1 <= len(args.symbol) <= 8 or not args.symbol.isascii():
        parser.error("symbol must contain 1..8 ASCII characters")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda day: fetch(day, args.output, args.symbol, args.scanner, args.ranged_download), args.day or list(DAYS)))


if __name__ == "__main__":
    main()
