"""Selected AI-Hub labels only; standard library, Windows/Linux.

Protocol reference: official aihubshell 25.09.19 v0.6.
No credentials on the command line or in logs. No automatic source deletion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlsplit, urljoin
import uuid
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.aihub.or.kr"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_hash(source, actual):
    lock_path = ROOT / "manifests/label_archives.json"
    if not lock_path.exists():
        return
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matches = [entry for entry in lock["sources"]
               if (entry["datasetkey"], entry["filekey"]) == (source["datasetkey"], source["filekey"])]
    if len(matches) != 1 or matches[0]["sha256"] != actual:
        raise ValueError(f"Published archive lock mismatch: filekey={source['filekey']}; do not substitute data")


def api_key():
    key = os.environ.get("AIHUB_API_KEY", "")
    if not key and os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
            try:
                key = winreg.QueryValueEx(registry, "AIHUB_API_KEY")[0]
            except FileNotFoundError:
                pass
    if not isinstance(key, str) or not key.strip():
        raise ValueError("AIHUB_API_KEY is missing. Set it locally; do not share it.")
    if key != key.strip() or "\r" in key or "\n" in key:
        raise ValueError("API key contains whitespace. Correct the local variable.")
    return key


class CurlResponse:
    """Use system curl like the official Shell; send credentials via stdin."""
    def __init__(self, url, key=None, redirects=0, max_bytes=2 * 1024**3):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in {"api.aihub.or.kr", "stream.aihub.or.kr"} or parsed.port not in (None, 443):
            raise ValueError("Only the official AI-Hub origin is allowed")
        if redirects > 3:
            raise ValueError("Too many AI-Hub redirects")
        self.url, self.key, self.redirects = url, key, redirects
        self.max_bytes = max_bytes
        self.redirect_response = None
        temp = ROOT / "data/transport"
        temp.mkdir(parents=True, exist_ok=True)
        self.header_path = temp / f"{uuid.uuid4().hex}.headers"
        command = ["curl.exe" if os.name == "nt" else "curl", "--silent", "--show-error",
                   "--connect-timeout", "30", "--max-time", "3600",
                   "--max-filesize", str(max_bytes), "--dump-header", str(self.header_path),
                   "--config", "-", url]
        config = ""
        if key:
            escaped = key.replace("\\", "\\\\").replace('"', '\\"')
            config = f'header = "apikey: {escaped}"\n'
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.process.stdin.write(config.encode("utf-8"))
        self.process.stdin.close()
        self.headers = {}

    def read(self, size=-1):
        if self.redirect_response:
            return self.redirect_response.read(size)
        chunk = self.process.stdout.read(size)
        if not chunk or size == -1:
            status = self.process.wait()
            if status:
                raise ValueError(f"AI-Hub curl request failed (exit {status}); response suppressed")
            lines = self.header_path.read_text(encoding="iso-8859-1").splitlines()
            location = next((line.split(":", 1)[1].strip() for line in reversed(lines)
                             if line.lower().startswith("location:")), None)
            if location:
                self.redirect_response = CurlResponse(urljoin(self.url, location), self.key, self.redirects + 1, self.max_bytes)
                return self.redirect_response.read(size)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.redirect_response:
            self.redirect_response.__exit__(*args)
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdout.close()
        self.process.stderr.close()
        self.header_path.unlink(missing_ok=True)


def request(url, key=None):
    return CurlResponse(url, key)


def load_sources():
    config = json.loads((ROOT / "configs/label_sources.json").read_text(encoding="utf-8"))
    sources = config["sources"]
    seen = set()
    for source in sources:
        pair = source["datasetkey"], source["filekey"]
        if not all(isinstance(n, int) and n > 0 for n in pair) or pair in seen:
            raise ValueError("Invalid or duplicate datasetkey/filekey")
        if source["datasetkey"] not in range(118, 123):
            raise ValueError("Unexpected dataset")
        if not source["filename"].endswith(".zip"):
            raise ValueError("Only explicitly selected label ZIPs are allowed")
        seen.add(pair)
    return config


def catalog(sources):
    dest = ROOT / "data/catalog"
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trees = {}
    for dataset in sorted({s["datasetkey"] for s in sources}):
        with request(f"{BASE}/info/{dataset}.do") as response:
            body = response.read().decode("utf-8-sig")
        (dest / f"{dataset}_{stamp}.txt").write_text(body, encoding="utf-8")
        trees[dataset] = body
    for source in sources:
        lines = [line for line in trees[source["datasetkey"]].splitlines()
                 if re.search(rf"\|\s*{source['filekey']}\s*$", line)]
        if len(lines) != 1 or source["filename"] not in lines[0]:
            raise ValueError(f"Catalog changed for filekey={source['filekey']}; review manifest")
        print(f"Verified catalog: {source['datasetkey']}/{source['filekey']} {source['display_size']}", flush=True)


def validate_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe archive member path")
    return path


def rebuild_zip(tar_path, output_path, expected_name):
    """Stream a selected ZIP out of tar without extracting arbitrary paths."""
    with tarfile.open(tar_path, "r:*") as archive:
        parts = []
        for member in archive.getmembers():
            path = validate_member(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Non-regular tar member")
            if path.name == expected_name:
                parts.append((-1, member))
            elif re.fullmatch(re.escape(expected_name) + r"\.part\d+", path.name):
                parts.append((int(path.name.rsplit(".part", 1)[1]), member))
            else:
                raise ValueError("Unexpected file in selected label download")
        offsets = [offset for offset, _ in parts]
        if not parts or len(set(offsets)) != len(parts) or (-1 in offsets and len(parts) != 1):
            raise ValueError("Missing or duplicate archive parts")
        partial = output_path.with_suffix(".zip.partial")
        with partial.open("wb") as out:
            for _, member in sorted(parts, key=lambda item: item[0]):
                with archive.extractfile(member) as stream:
                    shutil.copyfileobj(stream, out, 1024 * 1024)
        with zipfile.ZipFile(partial) as labels:
            for name in labels.namelist():
                validate_member(name)
            bad = labels.testzip()
            if bad:
                raise ValueError("ZIP CRC failed")
        partial.replace(output_path)


def download(source, version, key):
    directory = ROOT / "data/downloads" / str(source["datasetkey"]) / str(source["filekey"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "labels.zip"
    receipt_path = directory / "receipt.json"
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["sha256"] != sha256(target):
            raise ValueError("Cached ZIP hash mismatch; manual review required")
        verify_locked_hash(source, receipt["sha256"])
        print(f"Already verified: {source['filekey']}", flush=True)
        return
    tar_path = directory / "download.tar"
    if not tar_path.exists():
        if shutil.disk_usage(directory).free < 6 * 1024**3:
            raise ValueError("At least 6 GiB free space required for labels")
        url = f"{BASE}/down/{version}/{source['datasetkey']}.do?fileSn={source['filekey']}"
        partial = directory / "download.tar.partial"
        total, last = 0, time.monotonic()
        print(f"Downloading label filekey={source['filekey']}", flush=True)
        with request(url, key) as response, partial.open("wb") as out:
            content_type = response.headers.get("Content-Type", "").lower()
            if "json" in content_type or "text/" in content_type:
                raise ValueError("AI-Hub returned a non-archive response; check approval/API key")
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > 2 * 1024**3:
                    raise ValueError("Label download exceeds 2 GiB safety limit")
                out.write(chunk)
                if time.monotonic() - last > 10:
                    print(f"filekey={source['filekey']} received={total / 1024**2:.1f} MiB", flush=True)
                    last = time.monotonic()
            expected = response.headers.get("Content-Length")
            if expected and total != int(expected):
                raise ValueError("Incomplete HTTP body")
        if not tarfile.is_tarfile(partial):
            raise ValueError("Response is not tar; local response retained for review")
        partial.replace(tar_path)
    rebuild_zip(tar_path, target, source["filename"])
    actual_hash = sha256(target)
    verify_locked_hash(source, actual_hash)
    receipt = dict(source, api_version=version,
                   downloaded_at=datetime.now(timezone.utc).isoformat(),
                   bytes=target.stat().st_size, sha256=actual_hash,
                   transport_bytes=tar_path.stat().st_size, transport_sha256=sha256(tar_path))
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified ZIP and receipt: filekey={source['filekey']} bytes={receipt['bytes']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["catalog", "download"])
    parser.add_argument("--filekey", type=int, action="append", help="Optional subset of the label allowlist")
    args = parser.parse_args()
    config = load_sources()
    sources = config["sources"]
    if args.filekey:
        if set(args.filekey) - {s["filekey"] for s in sources}:
            parser.error("filekey not present in label allowlist")
        sources = [s for s in sources if s["filekey"] in args.filekey]
    catalog(sources)
    if args.action == "download":
        key = api_key()
        for source in sources:
            download(source, config["api_version"], key)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        sys.exit(1)
