"""Upload contract and restart behavior, without contacting a live service."""

import hashlib
import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import upload


UPLOAD_ID = "a" * 32
PROGRAM_ID = "b" * 20
API_KEY = "paw_sk_test_do_not_print"


@pytest.fixture
def exported(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "spec": "Classify support messages.", "runtime_id": "qwen3-0.6b-q6_k",
        "producer": "programasweights-compiler", "prompt_template": "Message: {INPUT_PLACEHOLDER}",
    }))
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen3-0.6B", "r": 8,
    }))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter-weights\x00" * 150000)
    return tmp_path


@pytest.fixture
def mock_server(monkeypatch):
    original_client = httpx.Client

    def install(handler):
        # BaseTransport preserves request chunks so the test observes streaming.
        class Transport(httpx.BaseTransport):
            def handle_request(self, request):
                return handler(request)

        monkeypatch.setattr(upload.httpx, "Client", lambda **kwargs: original_client(transport=Transport(), **kwargs))
    return install


def response(body, status=200):
    return httpx.Response(status, json=body)


def staged(status="staged", **fields):
    return {"upload_id": UPLOAD_ID, "status": status, **fields}


def registered(public=True):
    return {"program_id": PROGRAM_ID, "status": "registered", "created": True, "public": public}


def save_receipt(directory, **fields):
    state = {
        "api_url": upload.DEFAULT_API_URL, "public": True,
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in upload.FILES},
        "upload_id": UPLOAD_ID, **fields,
    }
    (directory / upload.RECEIPT).write_text(json.dumps(state))


def test_complete_upload_streams_weights_and_does_not_repeat_registration(exported, mock_server):
    calls = []
    weight_chunks = []

    def handle(request):
        calls.append((request.method, request.url.path))
        assert request.headers["X-API-Key"] == API_KEY
        assert request.url.host == "programasweights.com"
        if request.url.path == "/api/v1/program-imports":
            assert request.method == "POST"
            assert json.loads(request.read()) == {
                "manifest": json.loads((exported / "manifest.json").read_text()),
                "adapter_config": json.loads((exported / "adapter_config.json").read_text()),
            }
            return response(staged("awaiting_upload"), 201)
        if request.url.path.endswith("/weights"):
            assert request.method == "PUT"
            assert request.headers["Content-Type"] == "application/octet-stream"
            for chunk in request.stream:
                weight_chunks.append(chunk)
            assert b"".join(weight_chunks) == (exported / "adapter_model.safetensors").read_bytes()
            assert int(request.headers["Content-Length"]) == sum(map(len, weight_chunks))
            return response(staged())
        if request.url.path.endswith("/validate"):
            assert request.read() == b""
            return response(staged("validated", program_id=PROGRAM_ID))
        assert request.url.path.endswith("/register")
        assert json.loads(request.read()) == {"public": True}
        return response(registered(), 201)

    mock_server(handle)
    assert upload.upload_adapter(exported, api_key=API_KEY) == PROGRAM_ID
    assert len(calls) == 4
    assert len(weight_chunks) > 1
    assert max(map(len, weight_chunks)) <= 1024 * 1024
    assert upload.upload_adapter(exported, api_key=API_KEY) == PROGRAM_ID
    assert len(calls) == 4
    receipt = (exported / upload.RECEIPT).read_text()
    assert API_KEY not in receipt
    assert json.loads(receipt)["program_id"] == PROGRAM_ID


@pytest.mark.parametrize("status", ["staged", "failed", "validated"])
def test_resumes_saved_upload_without_reuploading_weights(exported, mock_server, status):
    save_receipt(exported)
    paths = []

    def handle(request):
        paths.append((request.method, request.url.path.rsplit("/", 1)[-1]))
        if request.method == "GET":
            return response(staged(status, program_id=PROGRAM_ID))
        if request.url.path.endswith("/validate"):
            return response(staged("validated", program_id=PROGRAM_ID))
        assert request.url.path.endswith("/register")
        return response(registered())

    mock_server(handle)
    assert upload.upload_adapter(exported, api_key=API_KEY) == PROGRAM_ID
    expected = [("GET", UPLOAD_ID)]
    if status != "validated":
        expected += [("POST", "validate")]
    assert paths == expected + [("POST", "register")]


def test_registration_timeout_resumes_same_upload(exported, mock_server):
    save_receipt(exported)
    registrations = []

    def handle(request):
        if request.method == "GET":
            return response(staged("validated", program_id=PROGRAM_ID))
        assert request.url.path == f"/api/v1/program-imports/{UPLOAD_ID}/register"
        registrations.append(json.loads(request.read()))
        if len(registrations) == 1:
            raise httpx.ReadTimeout("Response lost after registration", request=request)
        return response({**registered(), "created": False})

    mock_server(handle)
    with pytest.raises(httpx.ReadTimeout):
        upload.upload_adapter(exported, api_key=API_KEY)
    assert "program_id" not in json.loads((exported / upload.RECEIPT).read_text())
    assert upload.upload_adapter(exported, api_key=API_KEY) == PROGRAM_ID
    assert registrations == [{"public": True}, {"public": True}]


@pytest.mark.parametrize("http_status", [404, 410])
def test_missing_or_expired_upload_is_staged_again(exported, mock_server, http_status):
    save_receipt(exported)
    methods = []

    def handle(request):
        methods.append(request.method)
        if request.method == "GET":
            return response({"detail": "Upload expired"}, http_status)
        if request.url.path == "/api/v1/program-imports":
            return response(staged("awaiting_upload"), 201)
        if request.method == "PUT":
            request.read()
            return response(staged())
        if request.url.path.endswith("/validate"):
            return response(staged("validated", program_id=PROGRAM_ID))
        return response(registered())

    mock_server(handle)
    assert upload.upload_adapter(exported, api_key=API_KEY) == PROGRAM_ID
    assert methods == ["GET", "POST", "PUT", "POST", "POST"]


@pytest.mark.parametrize("http_status", [401, 403, 429, 503])
def test_status_error_does_not_create_another_upload(exported, mock_server, http_status):
    save_receipt(exported)
    calls = []

    def handle(request):
        calls.append(request.method)
        return response({"detail": "Retry later"}, http_status)

    mock_server(handle)
    with pytest.raises(upload.UploadError, match=f"HTTP {http_status}"):
        upload.upload_adapter(exported, api_key=API_KEY)
    assert calls == ["GET"]


@pytest.mark.parametrize("change", ["files", "server", "visibility"])
def test_receipt_binding_prevents_reusing_a_different_upload(exported, mock_server, change):
    save_receipt(exported, program_id=PROGRAM_ID)
    options = {}
    if change == "files":
        (exported / "adapter_model.safetensors").write_bytes(b"changed")
    elif change == "server":
        options["api_url"] = "http://localhost:8000"
    else:
        options["public"] = False

    def unexpected(request):
        pytest.fail("A mismatched receipt must not trigger a request")

    mock_server(unexpected)
    with pytest.raises(upload.UploadError, match="different files, visibility, or server"):
        upload.upload_adapter(exported, api_key=API_KEY, **options)


def test_registration_response_must_match_validated_program(exported, mock_server):
    save_receipt(exported)

    def handle(request):
        if request.method == "GET":
            return response(staged("validated", program_id=PROGRAM_ID))
        return response({**registered(), "program_id": "c" * 20})

    mock_server(handle)
    with pytest.raises(upload.UploadError, match="unexpected registration response"):
        upload.upload_adapter(exported, api_key=API_KEY)
    assert "program_id" not in json.loads((exported / upload.RECEIPT).read_text())


def test_cli_unlisted_upload_and_hub_link(exported, mock_server, monkeypatch, capsys):
    save_receipt(exported, public=False, api_url="http://localhost:8000")
    monkeypatch.setenv("PAW_API_KEY", API_KEY)

    def handle(request):
        assert request.url.host == "localhost"
        if request.method == "GET":
            return response(staged("validated", program_id=PROGRAM_ID))
        assert json.loads(request.read()) == {"public": False}
        return response(registered(public=False))

    mock_server(handle)
    assert upload.main([str(exported), "--unlisted", "--api-url", "http://localhost:8000/"]) == 0
    captured = capsys.readouterr()
    assert f"Program ID: {PROGRAM_ID}" in captured.out
    assert f"Hub: http://localhost:8000/hub/{PROGRAM_ID}" in captured.out
    assert API_KEY not in captured.out + captured.err


def test_cli_missing_key_is_actionable(exported, monkeypatch, capsys):
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    assert upload.main([str(exported)]) == 1
    assert "Set PAW_API_KEY" in capsys.readouterr().err


def test_cli_server_error_does_not_leak_api_key(exported, mock_server, monkeypatch, capsys):
    monkeypatch.setenv("PAW_API_KEY", API_KEY)
    mock_server(lambda request: response({"detail": "Bad key: " + API_KEY}, 401))
    assert upload.main([str(exported)]) == 1
    captured = capsys.readouterr()
    assert "HTTP 401" in captured.err
    assert API_KEY not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_cli_network_failure_keeps_receipt_and_prints_resume_instruction(exported, mock_server, monkeypatch, capsys):
    save_receipt(exported)
    monkeypatch.setenv("PAW_API_KEY", API_KEY)

    def fail(request):
        raise httpx.ConnectError("Network failure: " + API_KEY, request=request)

    mock_server(fail)
    assert upload.main([str(exported)]) == 1
    captured = capsys.readouterr()
    assert "Run the same command to resume" in captured.err
    assert API_KEY not in captured.out + captured.err
    assert json.loads((exported / upload.RECEIPT).read_text())["upload_id"] == UPLOAD_ID


@pytest.mark.parametrize("options", [["--api-url", "https://localhost:bad-port"],
                                     ["--api-url", "https://user:password@example.com"],
                                     ["--timeout", "nan"]])
def test_cli_invalid_options_do_not_produce_a_traceback(exported, monkeypatch, capsys, options):
    monkeypatch.setenv("PAW_API_KEY", API_KEY)
    assert upload.main([str(exported), *options]) == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_validation_failure_keeps_upload_for_retry(exported, mock_server):
    save_receipt(exported)
    attempts = []

    def handle(request):
        if request.method == "GET":
            return response(staged("staged" if not attempts else "failed"))
        if request.url.path.endswith("/validate"):
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return response({"detail": "Validation is unavailable; check upload status before retrying"}, 503)
            return response(staged("validated", program_id=PROGRAM_ID))
        assert request.url.path.endswith("/register")
        return response(registered())

    mock_server(handle)
    with pytest.raises(upload.UploadError, match="HTTP 503"):
        upload.upload_adapter(exported, api_key=API_KEY)
    assert upload.upload_adapter(exported, api_key=API_KEY) == PROGRAM_ID
    assert attempts == [f"/api/v1/program-imports/{UPLOAD_ID}/validate"] * 2
