#!/usr/bin/env python3
"""Offline plan by default. Explicit, owner-approved synthetic measurements only."""
import argparse
import base64
import hashlib
import http.client
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path


def png(size):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    row = b'\0' + bytes([32, 128, 224]) * size
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(row * size)) + chunk(b'IEND', b''))


def image(size, mime):
    data = png(size)
    if mime == 'image/jpeg':
        if not shutil.which('sips'):
            raise ValueError('JPEG fixtures require the macOS sips encoder; no customer files are accepted.')
        with tempfile.TemporaryDirectory(prefix='gbrain-synthetic-image-') as directory:
            source, target = Path(directory) / 'pixel.png', Path(directory) / 'pixel.jpg'
            source.write_bytes(data)
            subprocess.run(['sips', '-s', 'format', 'jpeg', str(source), '--out', str(target)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            data = target.read_bytes()
    return {'inline_data': {'mime_type': mime, 'data': base64.b64encode(data).decode()}}


def cases():
    rows = []
    def add(name, parts, **config):
        rows.append({'name': name, 'content': {'parts': parts},
                     'embedContentConfig': {'outputDimensionality': 768, 'autoTruncate': False, **config}})
    for name, text in [('english', 'Synthetic orange square.'), ('korean', '합성 주황 사각형.'),
                       ('ascii-8191', 'x' * 8191), ('ascii-8192', 'x' * 8192),
                       ('normalization', '\ufdfa' * 2730)]:
        add(name, [{'text': text}])
    for count in [1, 2, 8, 64, 1024]:
        add(f'parts-{count}', [{'text': 'a' * (2048 // count)} for _ in range(count)])
    add('parts-byte-ceiling', [{'text': 'a'}] * 8191)
    for truncate in [False, True]:
        add(f'overflow-{str(truncate).lower()}', [{'text': '1234567890' * 14000}], autoTruncate=truncate)
    add('overflow-default', [{'text': '1234567890' * 14000}])
    rows[-1]['embedContentConfig'].pop('autoTruncate')
    for dims in [127, 128, 3072]:
        add(f'dimensions-{dims}', [{'text': 'Synthetic orange square.'}], outputDimensionality=dims)
    add('invalid-config', [{'text': 'Synthetic orange square.'}], autoTruncate='invalid-boolean')
    for mime in ['image/png', 'image/jpeg']:
        for size in [1, 384, 385, 768, 1024]:
            part = image(size, mime)
            name = f'{mime.split("/")[1]}-{size}'
            add(name, [part])
            add(f'{name}-caption', [{'text': 'Synthetic orange square.'}, part])
        add(f'{mime.split("/")[1]}-six', [image(384, mime)] * 6)
        add(f'{mime.split("/")[1]}-overflow', [{'text': '1234567890' * 14000}, image(384, mime)])
    add('mixed-wire', [{'text': 'Synthetic orange square.'}, image(1, 'image/png')])
    add('camel-wire', [{'text': 'Synthetic orange square.'}, {'inlineData': {
        'mimeType': 'image/png', 'data': image(1, 'image/png')['inline_data']['data']}}])
    add('snake-config', [{'text': 'Synthetic orange square.'}])
    rows[-1]['embed_content_config'] = {'output_dimensionality': 768, 'auto_truncate': False}
    del rows[-1]['embedContentConfig']
    return rows


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(opener, endpoint, method, body, key):
    req = urllib.request.Request(endpoint + '/models/gemini-embedding-2:' + method,
                                 data=json.dumps(body).encode(), method='POST',
                                 headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    try:
        with opener.open(req, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, {}  # Never read or log a body that may echo credentials or input.
    except (OSError, ValueError, http.client.HTTPException):
        return None, {}  # Never print transport exception text or request objects.


def summary(status, data):
    data = data if isinstance(data, dict) else {}
    embedding = data.get('embedding')
    values = embedding.get('values') if isinstance(embedding, dict) else None
    usage = data.get('usageMetadata')
    usage = usage if isinstance(usage, dict) else {}
    detail_rows = usage.get('promptTokenDetails')
    detail_rows = detail_rows if isinstance(detail_rows, list) else []
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    details = [{'modality': row['modality'], 'tokenCount': row['tokenCount']}
               for row in detail_rows if isinstance(row, dict)
               and row.get('modality') in ['TEXT', 'IMAGE', 'AUDIO', 'VIDEO', 'DOCUMENT']
               and finite(row.get('tokenCount'))]
    return {'httpStatus': status, 'dimensions': len(values) if isinstance(values, list) else None,
            'finiteVector': bool(values) and all(finite(v) for v in values) if isinstance(values, list) else False,
            'vectorSha256': hashlib.sha256(json.dumps(values).encode()).hexdigest() if isinstance(values, list) else None,
            'promptTokenCount': usage.get('promptTokenCount') if finite(usage.get('promptTokenCount')) else None,
            'promptTokenDetails': details, 'countTokens': data.get('totalTokens') if finite(data.get('totalTokens')) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approval-ref')
    parser.add_argument('--endpoint', help='Exact HTTPS /v1beta endpoint approved by the owner; no default.')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--max-requests', type=int)
    args = parser.parse_args()
    if args.execute:
        url = urllib.parse.urlsplit(args.endpoint or '')
        if (not args.approval_ref or not args.case or not args.max_requests or url.scheme != 'https'
                or not url.hostname or url.username or url.password or url.query or url.fragment):
            parser.error('Execution requires approval-ref, selected cases, max-requests and a credential-free HTTPS endpoint.')
    rows = cases()
    if args.case:
        if set(args.case) - {row['name'] for row in rows}:
            parser.error('Unknown case name.')
        rows = [row for row in rows if row['name'] in args.case]
    if not args.execute:
        print(json.dumps({'mode': 'offline-plan', 'requestsIfExecuted': len(rows) * 2,
                          'cases': [row['name'] for row in rows]}))
        return
    if len(rows) * 2 > args.max_requests:
        parser.error('Selected cases exceed the approved request cap.')
    if sys.stdin.isatty():
        parser.error('Supply the credential through a protected stdin pipe, never a file or argument.')
    key = sys.stdin.read().strip()
    if not key:
        parser.error('No stdin credential; no request sent.')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    for row in rows:
        content = row['content']
        for method in ['countTokens', 'embedContent']:
            body = {'contents': [content]} if method == 'countTokens' else {k: v for k, v in row.items() if k != 'name'}
            status, data = request(opener, args.endpoint.rstrip('/'), method, body, key)
            print(json.dumps({'case': row['name'], 'method': method, **summary(status, data)}), flush=True)
            if status is None or 300 <= status < 400 or status in [401, 402, 403, 429] or (method == 'embedContent' and status == 404):
                raise SystemExit(1)  # No retries, alternate providers, redirects or key/region fallback.


if __name__ == '__main__':
    main()
