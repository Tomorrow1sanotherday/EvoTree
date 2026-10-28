"""Read-only request structure diagnostics; never persist images or prompts."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
from io import BytesIO


def request_metadata(kwargs):
    """Inspect what is passed to the client, not what a remote server received."""
    from PIL import Image

    messages = kwargs.get('messages') or []
    images = []
    text_parts = []
    roles = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        roles.append(message.get('role'))
        content = message.get('content', [])
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'text':
                text_parts.append(str(block.get('text', '')))
            elif block.get('type') == 'image_url':
                value = block.get('image_url') or {}
                url = value.get('url', '') if isinstance(value, dict) else ''
                row = dict(message=message_index, block=block_index,
                           inline=isinstance(url, str) and url.startswith('data:'),
                           valid_base64=False, decoded_image=False)
                if isinstance(url, str) and url.startswith('data:'):
                    try:
                        header, encoded = url.split(',', 1)
                        if ';base64' not in header:
                            raise ValueError('Image data URI is not base64')
                        raw = base64.b64decode(encoded, validate=True)
                        row.update(valid_base64=True, bytes=len(raw),
                                   sha256=hashlib.sha256(raw).hexdigest())
                        with Image.open(BytesIO(raw)) as bitmap:
                            row.update(width=bitmap.width, height=bitmap.height, format=bitmap.format)
                            bitmap.verify()
                        row['decoded_image'] = True
                    except (ValueError, binascii.Error, OSError, SyntaxError) as exc:
                        row['error_type'] = type(exc).__name__
                images.append(row)
    text = '\n'.join(text_parts)
    return dict(messages=len(messages), roles=roles, text_chars=len(text),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                xml_tool_instructions='<tools>' in text,
                native_tools=len(kwargs.get('tools') or []),
                image_count=len(images), images=images,
                model=kwargs.get('model'), max_tokens=kwargs.get('max_tokens'),
                temperature=kwargs.get('temperature'), top_p=kwargs.get('top_p'),
                scope='client-input-only; not evidence of remote receipt')
