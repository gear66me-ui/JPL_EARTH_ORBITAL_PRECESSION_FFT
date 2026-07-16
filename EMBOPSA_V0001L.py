# V0001L
# AUTOMATIC La2010 DATA DISCOVERY + FETCH
# NO AI-GENERATED IMAGES.

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from urllib.parse import quote

import ipywidgets as widgets
import pandas as pd
import requests
from IPython.display import display, clear_output

OUT = Path("/content/LA2010_DATA")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "JPL-EARTH-ORBITAL-PRECESSION-AUDIT/1.0",
}

results = []
state = {"selected": None}

search_button = widgets.Button(
    description="Find La2010 data",
    button_style="info",
    icon="search",
)

candidate_dd = widgets.Dropdown(
    options=[],
    description="Data file",
    layout=widgets.Layout(width="900px"),
)

download_button = widgets.Button(
    description="Fetch selected",
    button_style="success",
    icon="download",
    disabled=True,
)

status = widgets.HTML()
output = widgets.Output()

def request_json(url, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()

def add_candidate(source, title, filename, url, size=None, metadata=None):
    if not url:
        return
    key = url.strip()
    if any(x["url"] == key for x in results):
        return
    results.append({
        "source": source,
        "title": title,
        "filename": filename or Path(key.split("?")[0]).name or "download.dat",
        "url": key,
        "size": size,
        "metadata": metadata or {},
    })

def search_zenodo():
    queries = [
        '"La2010"',
        '"Laskar" "orbital solution"',
        '"ast5AL08cxc"',
        '"Earth orbital solution" "250 Myr"',
    ]
    for q in queries:
        try:
            data = request_json(
                "https://zenodo.org/api/records",
                params={"q": q, "size": 50, "sort": "bestmatch"},
            )
            for hit in data.get("hits", {}).get("hits", []):
                meta = hit.get("metadata", {})
                title = meta.get("title", "Zenodo record")
                for f in hit.get("files", []):
                    name = f.get("key", "")
                    lower = name.lower()
                    if lower.endswith((".txt", ".dat", ".csv", ".gz", ".zip")):
                        links = f.get("links", {})
                        add_candidate(
                            "Zenodo",
                            title,
                            name,
                            links.get("self") or links.get("download"),
                            f.get("size"),
                            {"record_id": hit.get("id")},
                        )
        except Exception:
            pass

def search_github_repositories():
    queries = [
        "La2010 orbital solution",
        "Laskar orbital Earth data",
        "ast5AL08cxc",
    ]
    for q in queries:
        try:
            data = request_json(
                "https://api.github.com/search/repositories",
                params={"q": q, "per_page": 20},
            )
            for repo in data.get("items", []):
                owner = repo["owner"]["login"]
                name = repo["name"]
                branch = repo.get("default_branch", "main")
                try:
                    tree = request_json(
                        f"https://api.github.com/repos/{owner}/{name}/git/trees/{branch}",
                        params={"recursive": "1"},
                    )
                except Exception:
                    continue

                for item in tree.get("tree", []):
                    path = item.get("path", "")
                    lower = path.lower()
                    match_name = any(term in lower for term in (
                        "la2010", "laskar", "ast5al08", "orbital"
                    ))
                    match_ext = lower.endswith((
                        ".txt", ".dat", ".csv", ".gz", ".zip"
                    ))
                    if match_name and match_ext:
                        raw = (
                            f"https://raw.githubusercontent.com/"
                            f"{owner}/{name}/{branch}/{quote(path)}"
                        )
                        add_candidate(
                            "GitHub",
                            repo.get("full_name", "GitHub repository"),
                            Path(path).name,
                            raw,
                            item.get("size"),
                            {"repository": repo.get("html_url"), "path": path},
                        )
        except Exception:
            pass

def search_github_code_with_token():
    token = None
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        pass
    if not token:
        return

    headers = dict(HEADERS)
    headers["Authorization"] = f"Bearer {token}"
    queries = [
        "La2010 extension:dat",
        "La2010 extension:txt",
        "ast5AL08cxc",
        "Laskar extension:csv",
    ]
    for q in queries:
        try:
            r = requests.get(
                "https://api.github.com/search/code",
                params={"q": q, "per_page": 50},
                headers=headers,
                timeout=60,
            )
            r.raise_for_status()
            for item in r.json().get("items", []):
                repo = item["repository"]["full_name"]
                path = item["path"]
                branch = item["repository"].get("default_branch", "main")
                raw = (
                    f"https://raw.githubusercontent.com/"
                    f"{repo}/{branch}/{quote(path)}"
                )
                add_candidate(
                    "GitHub code",
                    repo,
                    Path(path).name,
                    raw,
                    None,
                    {"path": path},
                )
        except Exception:
            pass

def looks_numeric(content):
    text = content[:200000].decode("utf-8", errors="ignore")
    lines = [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!", "%"))
    ]
    numeric = 0
    for line in lines[:500]:
        tokens = re.split(r"[\s,;]+", line)
        count = 0
        for token in tokens:
            try:
                float(token.replace("D", "E").replace("d", "e"))
                count += 1
            except Exception:
                pass
        if count >= 2:
            numeric += 1
    return numeric >= 5, numeric, len(lines)

def probe(candidate):
    try:
        r = requests.get(
            candidate["url"],
            headers=HEADERS,
            timeout=90,
            allow_redirects=True,
        )
        r.raise_for_status()
        content = r.content
        ctype = r.headers.get("content-type", "")
        binary_archive = (
            candidate["filename"].lower().endswith((".zip", ".gz"))
            or "zip" in ctype
            or "gzip" in ctype
        )
        if binary_archive:
            return True, "archive", len(content)
        ok, numeric, lines = looks_numeric(content)
        return ok, f"{numeric} numeric lines detected", len(content)
    except Exception as exc:
        return False, str(exc), 0

def run_search(_=None):
    global results
    results = []
    candidate_dd.options = []
    download_button.disabled = True
    status.value = "<b>Searching public archives…</b>"

    with output:
        clear_output(wait=True)

    search_zenodo()
    search_github_repositories()
    search_github_code_with_token()

    verified = []
    for candidate in results[:120]:
        ok, note, bytes_count = probe(candidate)
        if ok:
            candidate["probe"] = note
            candidate["bytes"] = bytes_count
            verified.append(candidate)

    results = verified

    if not results:
        status.value = (
            "<b style='color:#FF5C7A'>No verified downloadable La2010 "
            "numeric file was found automatically.</b><br>"
            "The finder checked Zenodo and public GitHub repositories. "
            "Adding a GITHUB_TOKEN Colab secret enables authenticated "
            "GitHub code search."
        )
        return

    options = []
    for i, item in enumerate(results):
        size = item.get("bytes") or item.get("size") or 0
        label = (
            f"{item['source']} | {item['filename']} | "
            f"{size/1_000_000:.2f} MB | {item['title']}"
        )
        options.append((label, i))

    candidate_dd.options = options
    candidate_dd.value = 0
    download_button.disabled = False
    status.value = f"<b>{len(results)} verified candidate file(s) found.</b>"
    show_selected()

def show_selected(*_):
    if not results or candidate_dd.value is None:
        return
    item = results[int(candidate_dd.value)]
    state["selected"] = item
    with output:
        clear_output(wait=True)
        display(pd.DataFrame([{
            "source": item["source"],
            "filename": item["filename"],
            "title": item["title"],
            "verified": item.get("probe"),
            "bytes": item.get("bytes"),
            "download_url": item["url"],
        }]))

def fetch_selected(_=None):
    item = state.get("selected")
    if not item:
        return
    try:
        status.value = "<b>Downloading and verifying…</b>"
        r = requests.get(
            item["url"],
            headers=HEADERS,
            timeout=180,
            allow_redirects=True,
        )
        r.raise_for_status()

        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", item["filename"])
        path = OUT / filename
        path.write_bytes(r.content)

        manifest = OUT / "LA2010_FETCH_MANIFEST.json"
        manifest.write_text(
            json.dumps({
                "source": item["source"],
                "title": item["title"],
                "filename": filename,
                "url": item["url"],
                "bytes": len(r.content),
                "metadata": item.get("metadata", {}),
            }, indent=2),
            encoding="utf-8",
        )

        status.value = (
            f"<b style='color:#35E0A1'>FETCHED:</b> "
            f"{path} ({len(r.content):,} bytes)"
        )
        with output:
            clear_output(wait=True)
            print(path)
            print(manifest)
    except Exception as exc:
        status.value = (
            f"<b style='color:#FF5C7A'>FETCH FAILED:</b> {exc}"
        )

search_button.on_click(run_search)
candidate_dd.observe(show_selected, names="value")
download_button.on_click(fetch_selected)

display(widgets.VBox([
    widgets.HBox([search_button, download_button]),
    candidate_dd,
    status,
    output,
]))
