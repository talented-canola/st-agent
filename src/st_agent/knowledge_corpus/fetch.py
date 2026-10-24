"""Anonymous, size-bounded, allowlisted external fetches with finite retries."""

from __future__ import annotations

import time
from urllib.parse import urljoin, urlsplit

import httpx

from .contracts import CorpusError, public_https

DEFAULT_HOSTS = frozenset(
    {
        "raw.githubusercontent.com",
        "www.ebi.ac.uk",
        "scanpy.readthedocs.io",
        "anndata.readthedocs.io",
    }
)


class HttpFetcher:
    def __init__(
        self,
        *,
        allowed_hosts=DEFAULT_HOSTS,
        transport=None,
        max_bytes=10_000_000,
        max_attempts=3,
        timeout=30,
        sleep=time.sleep,
        trace=None,
    ):
        if max_bytes < 1 or not 1 <= max_attempts <= 5:
            raise ValueError("Invalid fetch bounds")
        self.allowed_hosts = frozenset(allowed_hosts)
        self.max_bytes, self.max_attempts = max_bytes, max_attempts
        self.sleep, self.trace = sleep, trace
        self.request_count = 0
        self.client = httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "STAgent-ExternalCorpus/1"},
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def _validate(self, url):
        try:
            public_https(url)
        except ValueError as error:
            raise CorpusError("fetch.unsafe_url") from error
        if urlsplit(url).hostname not in self.allowed_hosts:
            raise CorpusError("fetch.host_not_allowed")

    def get(self, url):
        self._validate(url)
        for attempt in range(self.max_attempts):
            current = url
            try:
                for _ in range(6):
                    self._validate(current)
                    self.request_count += 1
                    if self.trace:
                        self.trace.emit("knowledge.fetch", url=current, attempt=attempt + 1)
                    with self.client.stream("GET", current) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            current = urljoin(current, response.headers.get("location", ""))
                            continue
                        if response.status_code in {429, 500, 502, 503, 504}:
                            raise CorpusError("fetch.transient_http_" + str(response.status_code))
                        if response.status_code != 200:
                            raise CorpusError("fetch.http_" + str(response.status_code))
                        output = bytearray()
                        for chunk in response.iter_bytes():
                            output.extend(chunk)
                            if len(output) > self.max_bytes:
                                raise CorpusError("fetch.response_too_large")
                        self.sleep(0.15)
                        return bytes(output)
                raise CorpusError("fetch.redirect_limit")
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                code = "fetch.network_" + type(error).__name__
            except CorpusError as error:
                code = str(error)
                if not code.startswith("fetch.transient_"):
                    raise
            if attempt + 1 == self.max_attempts:
                raise CorpusError(code)
            self.sleep(min(2**attempt, 3))
        raise CorpusError("fetch.attempts_exhausted")
