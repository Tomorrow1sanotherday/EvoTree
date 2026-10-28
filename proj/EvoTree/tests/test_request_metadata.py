import base64
import copy
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image

from evotree.request_metadata import request_metadata
from experiments.m2_small_eval.generalization_eval import RequestAudit


def test_image_structure_is_verified_without_logging_or_modifying_payload(tmp_path):
    bitmap = BytesIO()
    Image.new('RGB', (17, 13), (21, 43, 65)).save(bitmap, 'PNG')
    url = 'data:image/png;base64,' + base64.b64encode(bitmap.getvalue()).decode()
    kwargs = dict(model='model', messages=[dict(role='system', content='<tools>private tool schema</tools>'),
        dict(role='user', content=[dict(type='text', text='private task'),
                                  dict(type='image_url', image_url=dict(url=url))])])
    before = copy.deepcopy(kwargs)
    class Client:
        def create(self, **received):
            assert received == before
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                message=SimpleNamespace(content='ok', tool_calls=None))])
    path = tmp_path/'requests.jsonl'
    RequestAudit(Client(), path).create(**kwargs)
    assert kwargs == before
    saved = path.read_text()
    assert url not in saved and 'private task' not in saved and 'private tool schema' not in saved
    meta = json.loads(saved)['request_shape']
    assert meta['image_count'] == 1 and meta['xml_tool_instructions']
    assert meta['images'][0]['decoded_image']
    assert (meta['images'][0]['width'], meta['images'][0]['height']) == (17, 13)


def test_broken_base64_and_nonimage_bytes_are_distinguished():
    data = request_metadata(dict(messages=[dict(role='user', content=[
        dict(type='image_url', image_url=dict(url='data:image/png;base64,%%%')),
        dict(type='image_url', image_url=dict(url='data:image/png;base64,aGVsbG8='))])]))
    bad_base64, bad_image = data['images']
    assert not bad_base64['valid_base64']
    assert bad_image['valid_base64'] and not bad_image['decoded_image']


def test_backend_errors_still_record_request_structure(tmp_path):
    class Client:
        def create(self, **_):
            raise TimeoutError('remote failure')
    path = tmp_path/'requests.jsonl'
    try:
        RequestAudit(Client(),path).create(messages=[dict(role='user',content='private')])
    except TimeoutError:
        pass
    row = json.loads(path.read_text())
    assert row['error_type'] == 'TimeoutError' and row['request_shape']['messages'] == 1
    assert row['request_shape']['image_count'] == 0


def test_explicit_protocol_reminder_preserves_images_and_is_audited(tmp_path):
    from evotree.action_protocol import ACTION_PROTOCOL_REMINDER
    messages = [dict(role='user',content=[dict(type='text',text='original task'),
                    dict(type='image_url',image_url=dict(url='data:image/png;base64,aGVsbG8='))])]
    original = copy.deepcopy(messages)
    class Client:
        def create(self, **kwargs):
            assert kwargs['messages'][:-1] == original
            assert kwargs['messages'][-1] == dict(role='user',content=ACTION_PROTOCOL_REMINDER)
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='reply',tool_calls=None))])
    path=tmp_path/'requests.jsonl'
    RequestAudit(Client(),path,action_protocol_reminder=True).create(messages=messages)
    assert messages == original
    shape=json.loads(path.read_text())['request_shape']
    assert shape['messages']==2 and shape['image_count']==1
    assert shape['text_chars'] > len('original task')
