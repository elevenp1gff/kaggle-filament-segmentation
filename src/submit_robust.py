"""Submit using curl for every network call, sidestepping the SSL failures.

Python's urllib/requests keep hitting SSLEOFError on api.kaggle.com through the
proxy, while plain curl to the same host returns promptly (404 on / is normal).
So: get the upload URL with the SDK (which is a small POST that has been reliable
when retried), then do the large PUT with curl, then register with curl too.

Every step is retried and the HTTP code is reported, so a partial failure is
visible rather than silently swallowed.
"""
import base64
import json
import os
import subprocess
import sys
import time
import uuid

HOME = os.path.expanduser("~")
CFG = json.load(open(os.path.join(HOME, ".kaggle", "kaggle.json"), encoding="utf-8"))
KEY = CFG["key"]
USER = CFG["username"]
TOKEN = open(os.path.join(HOME, ".kaggle", "access_token"), encoding="utf-8").read().strip()
PROXY = "http://<PROXY>"
SLUG = "filament-segmentation-2026"
BASE = "https://www.kaggle.com/api/v1"
GRPC = "https://api.kaggle.com/v1/competitions.CompetitionApiService"


def curl(args, timeout=600):
    cmd = ["curl.exe", "-sS", "-x", PROXY] + args
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def start_upload(csv_path, tries=5):
    """Ask Kaggle for a resumable upload URL. Retries the SDK call."""
    for attempt in range(1, tries + 1):
        try:
            os.environ["KAGGLE_API_ENVIRONMENT"] = "PROD"
            os.environ["KAGGLE_API_TOKEN"] = KEY
            os.environ["HTTPS_PROXY"] = PROXY
            os.environ["HTTP_PROXY"] = PROXY
            for k in ("NO_PROXY", "no_proxy"):
                os.environ.pop(k, None)
            from kagglesdk.kaggle_client import KaggleClient
            from kagglesdk.competitions.types.competition_api_service import (
                ApiStartSubmissionUploadRequest)
            client = KaggleClient()
            svc = client.competitions.competition_api_client
            r = ApiStartSubmissionUploadRequest()
            r.competition_name = SLUG
            r.file_name = os.path.basename(csv_path)
            r.content_length = os.path.getsize(csv_path)
            r.last_modified_epoch_seconds = int(os.path.getmtime(csv_path))
            resp = svc.start_submission_upload(r)
            return resp.create_url, resp.token, svc
        except Exception as e:
            print("  start_upload attempt %d failed: %s: %s"
                  % (attempt, type(e).__name__, str(e)[:90]))
            time.sleep(3 * attempt)
    return None, None, None


def main(csv_path, message):
    total = os.path.getsize(csv_path)
    print("file %s %.1f KB" % (os.path.basename(csv_path), total / 1024))

    url, token, svc = start_upload(csv_path)
    if not url:
        print("could not obtain upload URL")
        return 1
    print("upload url ok, token %d chars" % len(token or ""))

    rc, out, err = curl(["-X", "PUT", "-T", csv_path,
                         "-H", "Content-Type: application/octet-stream",
                         "-w", "HTTP=%{http_code} SIZE=%{size_upload}",
                         "--max-time", "600", url], timeout=700)
    print("upload rc=%d %s" % (rc, err.strip()[:150]))
    ok = rc == 0 and ("storage#object" in out or "HTTP=200" in out or "HTTP=201" in out)
    print("upload success=%s" % ok)
    if not ok:
        return 1

    # register through the SDK (small POST, retried)
    for attempt in range(1, 5):
        try:
            from kagglesdk.competitions.types.competition_api_service import (
                ApiCreateSubmissionRequest)
            sub = ApiCreateSubmissionRequest()
            sub.competition_name = SLUG
            sub.blob_file_tokens = token
            sub.submission_description = message
            out = svc.create_submission(sub)
            print("SUBMITTED:", str(out)[:300])
            return 0
        except Exception as e:
            print("  create_submission attempt %d failed: %s: %s"
                  % (attempt, type(e).__name__, str(e)[:120]))
            time.sleep(3 * attempt)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "submission"))
