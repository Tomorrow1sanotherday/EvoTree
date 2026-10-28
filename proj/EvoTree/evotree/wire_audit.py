"""Opt-in local request/response capture without authentication headers.

This records the serialized SDK boundary, not what a remote model received.
Raw bodies may contain task text and screenshots; enable only for diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class WireAudit:
    def __init__(self, root, *, excluded_secrets=()):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.excluded_secrets = tuple(str(s).encode() for s in excluded_secrets if s)
        self.calls = 0
        self.records = []

    def _body(self, name, raw):
        if any(secret in raw for secret in self.excluded_secrets):
            raise ValueError('Credential-like content detected; refusing body capture')
        path = self.root / name
        with path.open('xb') as stream:
            os.chmod(path, 0o600)
            stream.write(raw)
        return dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def on_request(self, request):
        self.calls += 1
        body = request.read()
        record = dict(call=self.calls, method=request.method,
                      # Never inspect or persist request headers or URL query.
                      path=request.url.path,
                      request=self._body(f'{self.calls:04d}-request.json', body))
        self.records.append(record)
        request.extensions['evotree_wire_call'] = self.calls
        self._index()

    def on_response(self, response):
        call = response.request.extensions.get('evotree_wire_call')
        if call is None:
            raise ValueError('Response lacks matching audited request')
        raw = response.read()
        record = self.records[call - 1]
        record.update(status=response.status_code,
                      response=self._body(f'{call:04d}-response.json', raw))
        self._index()

    def _index(self):
        target = self.root / 'index.json'
        temp = self.root / 'index.tmp'
        with temp.open('w', encoding='utf8') as stream:
            os.chmod(temp, 0o600)
            json.dump(self.records, stream, indent=2)
        temp.replace(target)
