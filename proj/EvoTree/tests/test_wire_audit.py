import hashlib
import json

import httpx
import pytest

from evotree.wire_audit import WireAudit


def test_capture_keeps_exact_bodies_but_never_headers_or_query(tmp_path):
    root=tmp_path/'wire'
    audit=WireAudit(root, excluded_secrets=['secret-auth'])
    body=b'{ "messages": [{"content":"task"}], "image":"data:image/png;base64,aGVsbG8=" }'
    reply=b'{"choices":[{"message":{"content":"ok"}}]}'
    def serve(request):
        assert request.content==body
        return httpx.Response(200, content=reply, headers={'x-private':'secret-auth'})
    with httpx.Client(transport=httpx.MockTransport(serve),
                      event_hooks={'request':[audit.on_request],'response':[audit.on_response]}) as client:
        response=client.post('https://endpoint.invalid/v1/chat/completions?key=secret-auth',
                             content=body,headers={'Authorization':'Bearer secret-auth'})
    assert response.content==reply
    assert (root/'0001-request.json').read_bytes()==body
    assert (root/'0001-response.json').read_bytes()==reply
    assert 'secret-auth' not in ''.join(p.read_text() for p in root.iterdir())
    index=json.loads((root/'index.json').read_text())
    assert index[0]['request']['sha256']==hashlib.sha256(body).hexdigest()
    assert index[0]['path']=='/v1/chat/completions'


def test_capture_refuses_secret_in_a_body(tmp_path):
    audit=WireAudit(tmp_path/'wire',excluded_secrets=['hidden-key'])
    request=httpx.Request('POST','https://endpoint.invalid',content=b'{"value":"hidden-key"}')
    with pytest.raises(ValueError,match='refusing'):
        audit.on_request(request)
    assert not list((tmp_path/'wire').iterdir())


def test_existing_capture_is_never_overwritten(tmp_path):
    WireAudit(tmp_path/'wire')
    with pytest.raises(FileExistsError):WireAudit(tmp_path/'wire')
