"""Tests for preparing Balatro card-corner Label Studio imports."""

import json
from base64 import b64encode

import httpx
from PIL import Image

from ai_balatro.datasets.card_corner_label_studio import (
    build_label_config,
    build_tasks,
    create_project,
    import_tasks,
    refresh_access_token,
)


def test_build_tasks_maps_manifest_images_to_label_studio_urls(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(
        json.dumps(
            {
                'image': 'crops/out_00007-card-000.png',
                'source_image': 'images/out_00007.jpg',
                'source_label': 'labels/out_00007.txt',
                'slot_index': 0,
                'card_bbox_xyxy': [884, 324, 1038, 536],
                'corner_bbox_xyxy': [884, 324, 940, 422],
                'quality': {'light_ratio': 0.56, 'ink_ratio': 0.08},
            }
        )
        + '\n',
        encoding='utf-8',
    )

    tasks = build_tasks(
        manifest_path=manifest,
        image_url_prefix='http://nas.ihome.cat:10128/static/balatro-card-corners/crops',
    )

    assert tasks == [
        {
            'data': {
                'image': 'http://nas.ihome.cat:10128/static/balatro-card-corners/crops/out_00007-card-000.png',
                'crop_path': 'crops/out_00007-card-000.png',
                'source_image': 'images/out_00007.jpg',
                'source_label': 'labels/out_00007.txt',
                'slot_index': 0,
                'card_bbox_xyxy': [884, 324, 1038, 536],
                'corner_bbox_xyxy': [884, 324, 940, 422],
                'light_ratio': 0.56,
                'ink_ratio': 0.08,
            }
        }
    ]


def test_build_tasks_can_preserve_manifest_relative_path_under_prefix(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(
        json.dumps({'image': 'crops/out_00007-card-000.png'}) + '\n',
        encoding='utf-8',
    )

    tasks = build_tasks(
        manifest_path=manifest,
        image_url_prefix='http://nas.ihome.cat:10128/static/balatro-card-corners',
        image_url_mode='relative',
    )

    assert (
        tasks[0]['data']['image']
        == 'http://nas.ihome.cat:10128/static/balatro-card-corners/crops/out_00007-card-000.png'
    )


def test_build_tasks_can_embed_crop_images_as_data_uris(tmp_path):
    crops_dir = tmp_path / 'crops'
    crops_dir.mkdir()
    Image.new('RGB', (2, 1), 'red').save(crops_dir / 'sample.png')
    expected_png = (crops_dir / 'sample.png').read_bytes()

    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(
        json.dumps({'image': 'crops/sample.png'}) + '\n',
        encoding='utf-8',
    )

    tasks = build_tasks(
        manifest_path=manifest,
        embed_images=True,
    )

    assert tasks[0]['data']['image'] == (
        'data:image/png;base64,' + b64encode(expected_png).decode('ascii')
    )


def test_build_label_config_contains_rank_suit_and_unreadable_choices():
    label_config = build_label_config()

    assert '<Choice value="A"/>' in label_config
    assert '<Choice value="10"/>' in label_config
    assert '<Choice value="J" hotkey="j" />' in label_config
    assert '<Choice value="Q" hotkey="q" />' in label_config
    assert '<Choice value="K" hotkey="k" />' in label_config
    assert '<Choice value="spades" hotkey="s" />' in label_config
    assert '<Choice value="hearts" hotkey="h" />' in label_config
    assert '<Choice value="clubs" hotkey="c" />' in label_config
    assert '<Choice value="diamonds" hotkey="d" />' in label_config
    assert '<Choice value="readable" hotkey="y" />' in label_config
    assert '<Choice value="unreadable" hotkey="n" />' in label_config


def test_refresh_access_token_uses_label_studio_jwt_refresh_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == 'POST'
        assert request.url.path == '/api/token/refresh/'
        assert json.loads(request.content) == {'refresh': 'refresh-token'}
        return httpx.Response(200, json={'access': 'access-token'})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert (
        refresh_access_token(
            'http://labelstudio.local',
            'refresh-token',
            client=client,
        )
        == 'access-token'
    )


def test_create_project_and_import_tasks_use_bearer_access_token():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == '/api/projects':
            return httpx.Response(201, json={'id': 42, 'title': 'Balatro'})
        if request.url.path == '/api/projects/42/import':
            return httpx.Response(201, json={'task_count': 1})
        raise AssertionError(f'unexpected request path: {request.url.path}')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    project_id = create_project(
        'http://labelstudio.local',
        access_token='access-token',
        title='Balatro',
        label_config='<View/>',
        client=client,
    )
    result = import_tasks(
        'http://labelstudio.local',
        access_token='access-token',
        project_id=project_id,
        tasks=[{'data': {'image': 'http://example.test/crop.png'}}],
        client=client,
    )

    assert project_id == 42
    assert result == {'task_count': 1}
    assert requests[0].headers['authorization'] == 'Bearer access-token'
    assert requests[1].headers['authorization'] == 'Bearer access-token'
