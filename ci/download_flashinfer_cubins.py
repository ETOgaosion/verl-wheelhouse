#!/usr/bin/env python3
"""
Download flashinfer-cubin artifacts outside the wheel build step.

Uses a thread pool with HTTP keep-alive, skips files that already match
their checksum, and retries only failed downloads until the manifest is
complete. Intended to be run from ci/build_scripts/flashinfer.sh before
`python -m build` for flashinfer-cubin.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm


@dataclass(frozen=True)
class CubinArtifact:
    rel_name: str
    checksum: str
    url: str
    dest: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> bool:
    return path.is_file() and _sha256_file(path) == expected


def _download_one(
    session: requests.Session,
    artifact: CubinArtifact,
    *,
    retries: int,
    base_delay: float,
    timeout: float,
) -> bool:
    if _verify(artifact.dest, artifact.checksum):
        return True

    artifact.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact.dest.with_name(f"{artifact.dest.name}.{uuid.uuid4().hex}.tmp")

    for attempt in range(retries):
        try:
            response = session.get(artifact.url, timeout=timeout)
            response.raise_for_status()
            tmp.write_bytes(response.content)
            if not _verify(tmp, artifact.checksum):
                tmp.unlink(missing_ok=True)
                raise ValueError("checksum mismatch after download")
            os.replace(tmp, artifact.dest)
            return True
        except Exception as exc:  # noqa: BLE001 - collect and retry
            tmp.unlink(missing_ok=True)
            if attempt >= retries - 1:
                print(f"FAILED {artifact.rel_name}: {exc}", file=sys.stderr)
                return False
            cap = base_delay * (2**attempt)
            time.sleep(cap + random.uniform(0, cap))  # noqa: S311

    return False


def _make_session(workers: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _load_manifest(flashinfer_root: Path, cubin_dir: Path) -> list[CubinArtifact]:
    os.environ["FLASHINFER_CUBIN_DIR"] = str(cubin_dir)
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    root = str(flashinfer_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

    from flashinfer.artifacts import get_subdir_file_list
    from flashinfer.jit.cubin_loader import FLASHINFER_CUBINS_REPOSITORY, safe_urljoin

    return [
        CubinArtifact(
            rel_name=rel_name,
            checksum=checksum,
            url=safe_urljoin(FLASHINFER_CUBINS_REPOSITORY, rel_name),
            dest=cubin_dir / rel_name,
        )
        for rel_name, checksum in get_subdir_file_list()
    ]


def download_cubins(
    flashinfer_root: Path,
    cubin_dir: Path,
    *,
    workers: int,
    retries: int,
    timeout: float,
    max_rounds: int,
) -> None:
    cubin_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(flashinfer_root, cubin_dir)
    pending = manifest

    print(
        f"flashinfer cubin manifest: {len(manifest)} files -> {cubin_dir} "
        f"(workers={workers})"
    )

    for round_idx in range(1, max_rounds + 1):
        to_fetch = [item for item in pending if not _verify(item.dest, item.checksum)]
        if not to_fetch:
            print(f"All {len(pending)} cubin artifacts present and verified.")
            return

        print(f"Round {round_idx}/{max_rounds}: fetching {len(to_fetch)} file(s)...")
        failed: list[CubinArtifact] = []
        session = _make_session(workers)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _download_one,
                    session,
                    artifact,
                    retries=retries,
                    base_delay=2.0,
                    timeout=timeout,
                ): artifact
                for artifact in to_fetch
            }
            with tqdm(total=len(futures), desc="Downloading cubins", unit="file") as bar:
                for future in as_completed(futures):
                    artifact = futures[future]
                    if not future.result():
                        failed.append(artifact)
                    bar.update(1)

        if not failed:
            print(f"Round {round_idx}: all downloads succeeded.")
            return

        print(
            f"Round {round_idx}: {len(failed)} file(s) still missing; retrying...",
            file=sys.stderr,
        )
        pending = failed
        time.sleep(min(30.0, 5.0 * round_idx))

    names = ", ".join(item.rel_name for item in pending[:5])
    suffix = "" if len(pending) <= 5 else f", ... (+{len(pending) - 5} more)"
    raise SystemExit(
        f"Failed to download {len(pending)} cubin artifact(s) after {max_rounds} "
        f"rounds: {names}{suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the flashinfer submodule checkout (default: cwd)",
    )
    parser.add_argument(
        "--cubin-dir",
        type=Path,
        default=None,
        help="Destination cubin tree (default: <root>/flashinfer-cubin/flashinfer_cubin/cubins)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("FLASHINFER_CUBIN_DOWNLOAD_THREADS", "16")),
        help="Parallel download workers (default: 16, or FLASHINFER_CUBIN_DOWNLOAD_THREADS)",
    )
    parser.add_argument("--retries", type=int, default=8, help="Per-file HTTP retries")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Outer retry rounds for any files that still fail",
    )
    args = parser.parse_args()

    flashinfer_root = args.flashinfer_root.resolve()
    cubin_dir = args.cubin_dir or (
        flashinfer_root / "flashinfer-cubin" / "flashinfer_cubin" / "cubins"
    )

    download_cubins(
        flashinfer_root,
        cubin_dir.resolve(),
        workers=max(1, args.workers),
        retries=max(1, args.retries),
        timeout=args.timeout,
        max_rounds=max(1, args.max_rounds),
    )


if __name__ == "__main__":
    main()
