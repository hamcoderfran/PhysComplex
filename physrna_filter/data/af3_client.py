"""
AlphaFold 3 job builders and runners for PhysRNA predict-and-validate workflows.

Backends
--------
1. **server-json** — write AlphaFold Server JSON for upload at https://alphafoldserver.com
   (no public submit API; user uploads JSON or drops the zip back in).

2. **local** — run open-source AF3 via Docker (`run_alphafold.py`) when model/DB paths
   are configured (Linux/WSL + GPU).

3. **api** — experimental HTTP submit if ``AF3_API_KEY`` is set (endpoint may change).

4. **zip** — skip prediction; validate an existing AF3 Server download.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import requests

AF3_SERVER_DIALECT = "alphafoldserver"
AF3_SERVER_VERSION = 3

# Experimental — endpoint varies by provider; override with AF3_API_URL.
AF3_API_BASE = os.environ.get(
    "AF3_API_URL", "https://alphafoldserver.com/api"
).rstrip("/")
# Legacy EBI endpoint (some institutional keys)
AF3_EBI_API_URL = "https://api.alphafold.ebi.ac.uk/prediction"

_AA = set("ACDEFGHIKLMNPQRSTVWY")
_RNA = set("ACGU")


def validate_protein_sequence(sequence: str) -> str:
    seq = re.sub(r"\s+", "", sequence.upper())
    bad = sorted({c for c in seq if c not in _AA})
    if not seq:
        raise ValueError("Protein sequence is empty")
    if bad:
        raise ValueError(f"Invalid protein residue(s): {''.join(bad)}")
    return seq


def validate_rna_sequence(sequence: str) -> str:
    seq = re.sub(r"\s+", "", sequence.upper().replace("T", "U"))
    bad = sorted({c for c in seq if c not in _RNA})
    if not seq:
        raise ValueError("RNA sequence is empty")
    if bad:
        raise ValueError(f"Invalid RNA nucleotide(s): {''.join(bad)}")
    return seq


def build_alphafold_server_job(
    protein_sequence: str,
    rna_sequence: str,
    *,
    job_name: str = "physrna_fold",
    model_seeds: list[int] | None = None,
    protein_count: int = 1,
    rna_count: int = 1,
    use_structure_template: bool = True,
) -> dict:
    """Return one AlphaFold Server job dict (upload as JSON list of jobs)."""
    protein_sequence = validate_protein_sequence(protein_sequence)
    rna_sequence = validate_rna_sequence(rna_sequence)
    protein_chain: dict = {
        "sequence": protein_sequence,
        "count": protein_count,
    }
    if use_structure_template:
        protein_chain["useStructureTemplate"] = True
    return {
        "name": job_name,
        "modelSeeds": model_seeds if model_seeds is not None else [],
        "sequences": [
            {"proteinChain": protein_chain},
            {"rnaSequence": {"sequence": rna_sequence, "count": rna_count}},
        ],
        "dialect": AF3_SERVER_DIALECT,
        "version": AF3_SERVER_VERSION,
    }


def write_alphafold_server_json(
    protein_sequence: str,
    rna_sequence: str,
    output_path: str | Path,
    *,
    job_name: str = "physrna_fold",
) -> Path:
    """Write a Server-ready JSON file (top-level list with one job)."""
    job = build_alphafold_server_job(
        protein_sequence, rna_sequence, job_name=job_name
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([job], indent=2), encoding="utf-8")
    return out


def _af3_work_dir() -> Path:
    base = Path(os.environ.get("PHYRNA_AF3_WORK", Path.home() / ".cache" / "physrna_filter" / "af3_jobs"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_local_alphafold3(
    protein_sequence: str,
    rna_sequence: str,
    *,
    job_name: str = "physrna_fold",
    timeout_s: int = 7200,
) -> Path:
    """
    Run AF3 inference via Docker. Requires:

      AF3_MODEL_DIR, AF3_DB_DIR  — host paths
      AF3_DOCKER_IMAGE           — default ``alphafold3``
      AF3_DOCKER_BIN             — default ``docker`` (or ``wsl docker`` on Windows)
    """
    model_dir = os.environ.get("AF3_MODEL_DIR")
    db_dir = os.environ.get("AF3_DB_DIR")
    if not model_dir or not db_dir:
        raise EnvironmentError(
            "Local AF3 requires AF3_MODEL_DIR and AF3_DB_DIR environment variables."
        )
    if not Path(model_dir).is_dir() or not Path(db_dir).is_dir():
        raise FileNotFoundError("AF3_MODEL_DIR or AF3_DB_DIR path does not exist")

    docker = os.environ.get("AF3_DOCKER_BIN", "docker").strip()
    image = os.environ.get("AF3_DOCKER_IMAGE", "alphafold3")
    docker_cmd = docker.split()

    work = _af3_work_dir() / f"{job_name}_{uuid.uuid4().hex[:8]}"
    input_dir = work / "input"
    output_dir = work / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    json_path = input_dir / "fold_input.json"
    write_alphafold_server_json(
        protein_sequence, rna_sequence, json_path, job_name=job_name
    )

    gpu_flag = os.environ.get("AF3_USE_GPU", "1").lower() not in ("0", "false", "no")
    cmd = [
        *docker_cmd, "run", "--rm",
        "-v", f"{input_dir.resolve()}:/root/af_input",
        "-v", f"{output_dir.resolve()}:/root/af_output",
        "-v", f"{Path(model_dir).resolve()}:/root/models",
        "-v", f"{Path(db_dir).resolve()}:/root/public_databases",
    ]
    if gpu_flag:
        cmd.extend(["--gpus", "all"])
    cmd.extend([
        image,
        "python", "run_alphafold.py",
        f"--json_path=/root/af_input/{json_path.name}",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output",
    ])

    print("Running local AF3:", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout_s)

    zips = sorted(output_dir.rglob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if zips:
        return zips[0]
    cifs = sorted(output_dir.rglob("*.cif"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cifs:
        return cifs[0]
    raise FileNotFoundError(f"No AF3 output in {output_dir}")


def _resolve_api_key(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    from ..config import load_af3_api_key

    key = load_af3_api_key()
    if not key:
        raise EnvironmentError(
            "AlphaFold API key not set. Run:\n"
            "  physrna configure af3 --api-key YOUR_KEY\n"
            "or set AF3_API_KEY / ALPHAFOLD_API_KEY.\n"
            "Without API access, use --mode server-json and upload JSON at "
            "https://alphafoldserver.com"
        )
    return key


def _api_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _extract_job_id(data: object) -> str | None:
    if isinstance(data, dict):
        for key in ("jobId", "job_id", "id", "name"):
            val = data.get(key)
            if val:
                return str(val)
        for nested in ("job", "data", "result"):
            inner = data.get(nested)
            if isinstance(inner, dict):
                found = _extract_job_id(inner)
                if found:
                    return found
    if isinstance(data, list) and data:
        return _extract_job_id(data[0])
    return None


def _extract_status(data: object) -> str:
    if isinstance(data, dict):
        for key in ("status", "state", "jobStatus"):
            if data.get(key):
                return str(data[key]).upper()
    return ""


def _extract_download_urls(data: object) -> dict[str, str | None]:
    urls: dict[str, str | None] = {
        "zip_url": None,
        "structure_url": None,
        "ptm": None,
        "iptm": None,
    }
    if not isinstance(data, dict):
        return urls

    for zip_key in ("zip_url", "zipUrl", "download_url", "downloadUrl", "archiveUrl"):
        if data.get(zip_key):
            urls["zip_url"] = str(data[zip_key])
            break
    for struct_key in ("structure_url", "structureUrl", "pdb_url", "cif_url", "modelUrl"):
        if data.get(struct_key):
            urls["structure_url"] = str(data[struct_key])
            break
    if data.get("ptm") is not None:
        urls["ptm"] = data.get("ptm")
    if data.get("iptm") is not None:
        urls["iptm"] = data.get("iptm")

    for nested in ("result", "output", "artifacts", "downloads"):
        inner = data.get(nested)
        if isinstance(inner, dict):
            sub = _extract_download_urls(inner)
            for k, v in sub.items():
                if v and not urls.get(k):
                    urls[k] = v
    return urls


def _api_poll_interval_s() -> int:
    raw = os.environ.get("AF3_API_POLL_S", "30").strip()
    try:
        return max(5, int(raw))
    except ValueError:
        return 30


def _submit_job_requests(
    job: dict,
    api_key: str,
    *,
    timeout_s: int = 120,
) -> tuple[str, dict, str]:
    """
    Try common AlphaFold HTTP submit patterns.

    Returns (job_id, raw_response, submit_url_used).
    """
    headers = _api_headers(api_key)
    payloads = [job, [job]]
    submit_urls = [
        f"{AF3_API_BASE}/submit",
        f"{AF3_API_BASE}/predictions",
        f"{AF3_API_BASE}/jobs",
        AF3_API_BASE,
        AF3_EBI_API_URL,
    ]
    # De-duplicate while preserving order
    seen: set[str] = set()
    submit_urls = [u for u in submit_urls if not (u in seen or seen.add(u))]

    errors: list[str] = []
    for url in submit_urls:
        for payload in payloads:
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
                if resp.status_code in (404, 405):
                    errors.append(f"{url}: HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                job_id = _extract_job_id(data)
                if job_id:
                    return job_id, data if isinstance(data, dict) else {"raw": data}, url
                errors.append(f"{url}: no job id in {data!r}")
            except requests.RequestException as exc:
                errors.append(f"{url}: {exc}")
            except ValueError as exc:
                errors.append(f"{url}: invalid JSON ({exc})")

    raise RuntimeError(
        "AlphaFold API submit failed on all known endpoints.\n"
        + "\n".join(f"  - {e}" for e in errors[:6])
        + "\nTry --mode server-json, or set AF3_API_URL to your provider's base URL."
    )


def _poll_job_status(
    job_id: str,
    api_key: str,
    *,
    submit_url: str | None = None,
    timeout_s: int = 3600,
) -> dict:
    headers = _api_headers(api_key)
    poll_urls = [
        f"{AF3_API_BASE}/job/{job_id}",
        f"{AF3_API_BASE}/jobs/{job_id}",
        f"{AF3_API_BASE}/predictions/{job_id}",
        f"{AF3_EBI_API_URL.rstrip('/')}/{job_id}",
    ]
    if submit_url:
        base = submit_url.rstrip("/")
        poll_urls.insert(0, f"{base}/{job_id}")

    interval = _api_poll_interval_s()
    t0 = time.time()
    last_state = ""
    while time.time() - t0 < timeout_s:
        for url in poll_urls:
            try:
                resp = requests.get(url, headers=headers, timeout=60)
                if resp.status_code in (404, 405):
                    continue
                resp.raise_for_status()
                status = resp.json()
                if isinstance(status, list) and status:
                    status = status[0]
                if not isinstance(status, dict):
                    continue
                state = _extract_status(status)
                if state and state != last_state:
                    print(f"  AF3 job {job_id}: {state}")
                    last_state = state
                if state in ("COMPLETED", "COMPLETE", "SUCCESS", "DONE", "SUCCEEDED"):
                    downloads = _extract_download_urls(status)
                    return {
                        "job_id": job_id,
                        "zip_url": downloads["zip_url"],
                        "structure_url": downloads["structure_url"],
                        "ptm": downloads["ptm"],
                        "iptm": downloads["iptm"],
                        "raw": status,
                    }
                if state in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
                    raise RuntimeError(f"AF3 API job failed: {status}")
            except requests.RequestException:
                continue
        time.sleep(interval)

    raise TimeoutError(f"AF3 API job {job_id} timed out after {timeout_s}s")


def submit_af3_api(
    protein_sequence: str,
    rna_sequence: str,
    *,
    job_name: str = "physrna_fold",
    api_key: str | None = None,
    timeout_s: int = 3600,
) -> dict:
    """
    Submit an AlphaFold job via HTTP API, poll until complete, return download URLs.

  Uses ``AF3_API_KEY``, ``ALPHAFOLD_API_KEY``, or ``~/.physrna/af3_api_key``.
    """
    key = _resolve_api_key(api_key)

    job = build_alphafold_server_job(
        protein_sequence, rna_sequence, job_name=job_name
    )
    job_id, _raw, submit_url = _submit_job_requests(job, key)
    print(f"Submitted AF3 job {job_id}")
    return _poll_job_status(job_id, key, submit_url=submit_url, timeout_s=timeout_s)


def resume_af3_api_job(
    job_id: str,
    *,
    api_key: str | None = None,
    timeout_s: int = 3600,
) -> dict:
    """Poll an existing AF3 API job and return download URLs when complete."""
    key = _resolve_api_key(api_key)
    print(f"Resuming AF3 job {job_id}")
    return _poll_job_status(job_id, key, timeout_s=timeout_s)


def download_url_to_file(url: str, dest: Path, timeout_s: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return dest


def resolve_backend(mode: str) -> str:
    """Pick a backend when mode=auto."""
    if mode != "auto":
        return mode
    if os.environ.get("AF3_MODEL_DIR") and os.environ.get("AF3_DB_DIR"):
        return "local"
    from ..config import load_af3_api_key

    if load_af3_api_key():
        return "api"
    return "server-json"


def predict_structure(
    protein_sequence: str,
    rna_sequence: str,
    *,
    mode: str = "auto",
    job_name: str = "physrna_fold",
    server_json_path: str | Path | None = None,
    af3_zip: str | Path | None = None,
    api_key: str | None = None,
    af3_job_id: str | None = None,
    timeout_s: int = 7200,
) -> dict:
    """
    Obtain a structure path for PhysRNA validation.

    Returns dict with keys: backend, structure_path, server_json, job_name, notes.
    """
    backend = resolve_backend(mode)
    notes: list[str] = []

    if af3_zip:
        p = Path(af3_zip)
        if not p.exists():
            raise FileNotFoundError(af3_zip)
        return {
            "backend": "zip",
            "structure_path": str(p.resolve()),
            "server_json": None,
            "job_name": job_name,
            "notes": ["Using provided AF3 zip/structure"],
        }

    if backend == "zip":
        raise ValueError("mode=zip requires --af3-zip PATH")

    if backend == "server-json":
        out = Path(server_json_path) if server_json_path else _af3_work_dir() / f"{job_name}.json"
        write_alphafold_server_json(
            protein_sequence, rna_sequence, out, job_name=job_name
        )
        notes.append(
            "Upload the JSON at https://alphafoldserver.com (Upload JSON), "
            "run the job, download the zip, then re-run with --af3-zip fold_<name>.zip"
        )
        return {
            "backend": "server-json",
            "structure_path": None,
            "server_json": str(out.resolve()),
            "job_name": job_name,
            "notes": notes,
        }

    if backend == "local":
        artifact = run_local_alphafold3(
            protein_sequence, rna_sequence, job_name=job_name, timeout_s=timeout_s
        )
        return {
            "backend": "local",
            "structure_path": str(artifact.resolve()),
            "server_json": None,
            "job_name": job_name,
            "notes": ["Local Docker AF3 completed"],
        }

    if backend == "api":
        if af3_job_id:
            result = resume_af3_api_job(
                af3_job_id, api_key=api_key, timeout_s=timeout_s
            )
        else:
            result = submit_af3_api(
                protein_sequence,
                rna_sequence,
                job_name=job_name,
                api_key=api_key,
                timeout_s=timeout_s,
            )
        structure_path = None
        if result.get("zip_url"):
            dest = _af3_work_dir() / f"{job_name}_{result['job_id']}.zip"
            download_url_to_file(result["zip_url"], dest)
            structure_path = str(dest)
        elif result.get("structure_url"):
            dest = _af3_work_dir() / f"{job_name}_{result['job_id']}.cif"
            download_url_to_file(result["structure_url"], dest)
            structure_path = str(dest)
        return {
            "backend": "api",
            "structure_path": structure_path,
            "server_json": None,
            "job_name": job_name,
            "notes": [f"AF3 API job {result.get('job_id')}"],
            "af3_confidence": {
                "ptm": result.get("ptm"),
                "iptm": result.get("iptm"),
            },
        }

    raise ValueError(f"Unknown AF3 mode: {backend}")
