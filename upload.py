#!/usr/bin/env python3
"""Upload an exported compiler adapter to ProgramAsWeights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import httpx


DEFAULT_API_URL = "https://programasweights.com"
FILES = ("manifest.json", "adapter_config.json", "adapter_model.safetensors")
RECEIPT = ".paw-upload.json"


class UploadError(Exception):
    """An upload error that can be shown directly to the user."""


def _api_url(value):
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL as exc:
        raise UploadError("Use a valid HTTP(S) server URL.") from exc
    if (url.scheme not in {"http", "https"} or not url.host
            or url.userinfo or url.query or url.fragment):
        raise UploadError("Use an HTTP(S) server URL without credentials, a query, or a fragment.")
    return str(url).rstrip("/")


def _json_file(path):
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise UploadError(f"{path.name} must contain valid JSON.") from exc
    if not isinstance(value, dict):
        raise UploadError(f"{path.name} must contain a JSON object.")
    return value


def _save_receipt(path, state):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(state, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _response_json(response, api_key):
    if not response.is_success:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        message = detail if isinstance(detail, str) else response.reason_phrase
        raise UploadError(f"Upload failed (HTTP {response.status_code}): {message}".replace(api_key, "[redacted]"))
    try:
        data = response.json()
    except ValueError as exc:
        raise UploadError("The server returned an invalid upload response. Run the same command to resume.") from exc
    if not isinstance(data, dict):
        raise UploadError("The server returned an invalid upload response. Run the same command to resume.")
    return data


def _id(value, length, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
        raise UploadError(f"The server returned an invalid {name}. Run the same command to resume.")
    return value


def upload_adapter(directory, *, api_key, api_url=DEFAULT_API_URL, public=True, timeout=600):
    """Upload three exported files, resuming the same import on subsequent calls."""
    if not api_key:
        raise UploadError("Set PAW_API_KEY to your ProgramAsWeights API key.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise UploadError("The timeout must be a finite number greater than zero.")
    directory, api_url = Path(directory), _api_url(api_url)
    fingerprints = {}
    for name in FILES:
        path = directory / name
        if not path.is_file():
            raise UploadError(f"Missing {name} in {directory}.")
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprints[name] = digest.hexdigest()
    metadata = {"manifest": _json_file(directory / "manifest.json"),
                "adapter_config": _json_file(directory / "adapter_config.json")}
    if not (directory / "adapter_model.safetensors").stat().st_size:
        raise UploadError("adapter_model.safetensors must not be empty.")

    receipt_path = directory / RECEIPT
    binding = {"api_url": api_url, "public": public, "files": fingerprints}
    state = _json_file(receipt_path) if receipt_path.exists() else binding.copy()
    if any(state.get(key) != value for key, value in binding.items()):
        raise UploadError("The saved upload uses different files, visibility, or server. "
                          "Export to a new directory for a separate upload.")
    if state.get("program_id"):
        return _id(state["program_id"], 20, "program ID")

    with httpx.Client(base_url=api_url + "/api/v1/", headers={"X-API-Key": api_key}, timeout=timeout) as client:
        def request(method, path, **kwargs):
            return _response_json(client.request(method, path, **kwargs), api_key)

        status = None
        if state.get("upload_id"):
            upload_id = _id(state["upload_id"], 32, "upload ID")
            response = client.get(f"program-imports/{upload_id}")
            if response.status_code not in {404, 410}:
                status = _response_json(response, api_key)
        if status is None:
            status = request("POST", "program-imports", json=metadata)
            state["upload_id"] = _id(status.get("upload_id"), 32, "upload ID")
            _save_receipt(receipt_path, state)
        path = f"program-imports/{state['upload_id']}"
        if status.get("status") == "awaiting_upload":
            weights_path = directory / "adapter_model.safetensors"
            with weights_path.open("rb") as weights:
                status = request("PUT", path + "/weights",
                                 content=iter(lambda: weights.read(1024 * 1024), b""),
                                 headers={"Content-Type": "application/octet-stream",
                                          "Content-Length": str(weights_path.stat().st_size)})
        if status.get("status") in {"staged", "failed"}:
            status = request("POST", path + "/validate")
        if status.get("status") != "validated":
            raise UploadError("The adapter is not validated yet. Run the same command to resume.")
        program_id = _id(status.get("program_id"), 20, "program ID")
        registered = request("POST", path + "/register", json={"public": public})
        if (registered.get("status") != "registered" or registered.get("program_id") != program_id
                or registered.get("public") is not public):
            raise UploadError("The server returned an unexpected registration response. "
                              "Run the same command to check registration.")
        state["program_id"] = program_id
        _save_receipt(receipt_path, state)
        return program_id


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload an exported adapter to ProgramAsWeights. "
                                                 "Programs are listed publicly by default.")
    parser.add_argument("directory", type=Path, help="directory containing the exported adapter and manifest")
    parser.add_argument("--unlisted", action="store_true",
                        help="hide the program from public listings; anyone with its ID can still use it")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="server URL (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=600,
                        help="timeout in seconds per request (default: %(default)s)")
    args = parser.parse_args(argv)
    api_key = os.environ.get("PAW_API_KEY", "")
    try:
        program_id = upload_adapter(args.directory, api_key=api_key, api_url=args.api_url,
                                    public=not args.unlisted, timeout=args.timeout)
    except httpx.HTTPError:
        print("Upload interrupted while contacting the server. Run the same command to resume.", file=sys.stderr)
        return 1
    except (UploadError, OSError, ValueError) as exc:
        message = str(exc).replace(api_key, "[redacted]") if api_key else str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 1
    print(f"Program ID: {program_id}")
    print(f"Hub: {_api_url(args.api_url)}/hub/{program_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
