"""Prepare and import Balatro card-corner crops into Label Studio."""

from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


def build_label_config() -> str:
    """Build a Label Studio config for card rank/suit classification."""

    return """<View>
  <Image name="image" value="$image"/>

  <Header value="Rank"/>
  <Choices name="rank" toName="image" choice="single" showInline="true">
    <Choice value="A"/>
    <Choice value="2"/>
    <Choice value="3"/>
    <Choice value="4"/>
    <Choice value="5"/>
    <Choice value="6"/>
    <Choice value="7"/>
    <Choice value="8"/>
    <Choice value="9"/>
    <Choice value="10"/>
    <Choice value="J" hotkey="j" />
    <Choice value="Q" hotkey="q" />
    <Choice value="K" hotkey="k" />
  </Choices>

  <Header value="Suit"/>
  <Choices name="suit" toName="image" choice="single" showInline="true">
    <Choice value="spades" hotkey="s" />
    <Choice value="hearts" hotkey="h" />
    <Choice value="clubs" hotkey="c" />
    <Choice value="diamonds" hotkey="d" />
  </Choices>

  <Header value="Quality"/>
  <Choices name="quality" toName="image" choice="single" showInline="true">
    <Choice value="readable" hotkey="y" />
    <Choice value="unreadable" hotkey="n" />
  </Choices>
</View>"""


def build_tasks(
    *,
    manifest_path: Path,
    image_url_prefix: str | None = None,
    image_url_mode: str = 'basename',
    embed_images: bool = False,
    crop_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build Label Studio import tasks from a card-corner manifest."""

    tasks = []
    for sample in _read_manifest(manifest_path):
        image_path = sample['image']
        quality = sample.get('quality', {})
        tasks.append(
            {
                'data': {
                    'image': _build_image_value(
                        image_path=image_path,
                        manifest_path=manifest_path,
                        image_url_prefix=image_url_prefix,
                        image_url_mode=image_url_mode,
                        embed_images=embed_images,
                        crop_root=crop_root,
                    ),
                    'crop_path': image_path,
                    'source_image': sample.get('source_image'),
                    'source_label': sample.get('source_label'),
                    'slot_index': sample.get('slot_index'),
                    'card_bbox_xyxy': sample.get('card_bbox_xyxy'),
                    'corner_bbox_xyxy': sample.get('corner_bbox_xyxy'),
                    'light_ratio': quality.get('light_ratio'),
                    'ink_ratio': quality.get('ink_ratio'),
                }
            }
        )

    return tasks


def write_label_studio_files(
    *,
    manifest_path: Path,
    out_dir: Path,
    image_url_prefix: str | None = None,
    image_url_mode: str = 'basename',
    embed_images: bool = False,
    crop_root: Path | None = None,
) -> tuple[Path, Path]:
    """Write Label Studio tasks JSON and label config XML."""

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / 'label-studio-tasks.json'
    config_path = out_dir / 'label-config.xml'

    tasks = build_tasks(
        manifest_path=manifest_path,
        image_url_prefix=image_url_prefix,
        image_url_mode=image_url_mode,
        embed_images=embed_images,
        crop_root=crop_root,
    )
    tasks_path.write_text(json.dumps(tasks, indent=2) + '\n', encoding='utf-8')
    config_path.write_text(build_label_config() + '\n', encoding='utf-8')

    return tasks_path, config_path


def refresh_access_token(
    label_studio_url: str,
    refresh_token: str,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Exchange a Label Studio JWT refresh token for an access token."""

    own_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.post(
            _api_url(label_studio_url, '/api/token/refresh/'),
            json={'refresh': refresh_token},
        )
        response.raise_for_status()
        access_token = response.json().get('access')
        if not access_token:
            raise ValueError(
                'Label Studio token refresh response did not include access'
            )
        return access_token
    finally:
        if own_client:
            client.close()


def create_project(
    label_studio_url: str,
    *,
    access_token: str,
    title: str,
    label_config: str,
    client: httpx.Client | None = None,
) -> int:
    """Create a Label Studio project and return its project ID."""

    own_client = client is None
    client = client or httpx.Client(timeout=30)
    try:
        response = client.post(
            _api_url(label_studio_url, '/api/projects'),
            headers=_bearer_headers(access_token),
            json={'title': title, 'label_config': label_config},
        )
        response.raise_for_status()
        project_id = response.json().get('id')
        if project_id is None:
            raise ValueError(
                'Label Studio project creation response did not include id'
            )
        return int(project_id)
    finally:
        if own_client:
            client.close()


def import_tasks(
    label_studio_url: str,
    *,
    access_token: str,
    project_id: int,
    tasks: list[dict[str, Any]],
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Import tasks into an existing Label Studio project."""

    own_client = client is None
    client = client or httpx.Client(timeout=60)
    try:
        response = client.post(
            _api_url(label_studio_url, f'/api/projects/{project_id}/import'),
            headers=_bearer_headers(access_token),
            json=tasks,
        )
        response.raise_for_status()
        return response.json()
    finally:
        if own_client:
            client.close()


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    samples = []
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            samples.append(json.loads(line))
    return samples


def _build_image_value(
    *,
    image_path: str,
    manifest_path: Path,
    image_url_prefix: str | None,
    image_url_mode: str,
    embed_images: bool,
    crop_root: Path | None,
) -> str:
    if embed_images:
        root = crop_root or manifest_path.parent
        return _image_data_uri(root / image_path)

    if image_url_prefix is None:
        return image_path

    if image_url_mode == 'basename':
        suffix = Path(image_path).name
    elif image_url_mode == 'relative':
        suffix = image_path
    else:
        raise ValueError("image_url_mode must be 'basename' or 'relative'")

    return f'{image_url_prefix.rstrip("/")}/{quote(suffix)}'


def _image_data_uri(image_path: Path) -> str:
    media_type = mimetypes.guess_type(image_path.name)[0] or 'application/octet-stream'
    encoded = b64encode(image_path.read_bytes()).decode('ascii')
    return f'data:{media_type};base64,{encoded}'


def _api_url(label_studio_url: str, path: str) -> str:
    return f'{label_studio_url.rstrip("/")}{path}'


def _bearer_headers(access_token: str) -> dict[str, str]:
    return {'Authorization': f'Bearer {access_token}'}
