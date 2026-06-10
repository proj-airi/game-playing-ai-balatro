#!/usr/bin/env python3
"""Create/import Balatro card-corner tasks for Label Studio."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_cli_dir = Path(__file__).resolve().parent
_src_dir = _cli_dir.parent.parent / 'src'
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ai_balatro.datasets.card_corner_label_studio import (  # noqa: E402
    build_label_config,
    build_tasks,
    create_project,
    import_tasks,
    refresh_access_token,
    write_label_studio_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Import card-corner crop tasks into Label Studio',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out-dir', default=None, type=Path)
    parser.add_argument('--image-url-prefix', default=None)
    parser.add_argument(
        '--image-url-mode',
        choices=['basename', 'relative'],
        default='basename',
        help='Use only crop basenames or preserve manifest-relative image paths',
    )
    parser.add_argument(
        '--embed-images',
        action='store_true',
        help='Embed crop files as base64 data URIs so Label Studio needs no static file server',
    )
    parser.add_argument(
        '--crop-root',
        default=None,
        type=Path,
        help='Root for manifest-relative crop paths when using --embed-images',
    )
    parser.add_argument(
        '--label-studio-url',
        default=os.getenv('LABEL_STUDIO_URL', 'http://nas.ihome.cat:10128'),
    )
    parser.add_argument(
        '--refresh-token',
        default=os.getenv('LABEL_STUDIO_REFRESH_TOKEN'),
        help='JWT refresh token; exchanged for a Bearer access token',
    )
    parser.add_argument(
        '--access-token',
        default=os.getenv('LABEL_STUDIO_ACCESS_TOKEN'),
        help='JWT access token; avoids refresh when already available',
    )
    parser.add_argument('--project-id', default=None, type=int)
    parser.add_argument('--project-title', default='Balatro Card Corners')
    parser.add_argument(
        '--create-project',
        action='store_true',
        help='Create a Label Studio project before importing tasks',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only write tasks/config files; do not call Label Studio',
    )
    args = parser.parse_args()

    if args.out_dir is not None:
        tasks_path, config_path = write_label_studio_files(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            image_url_prefix=args.image_url_prefix,
            image_url_mode=args.image_url_mode,
            embed_images=args.embed_images,
            crop_root=args.crop_root,
        )
        print(f'Wrote tasks: {tasks_path}')
        print(f'Wrote label config: {config_path}')

    tasks = build_tasks(
        manifest_path=args.manifest,
        image_url_prefix=args.image_url_prefix,
        image_url_mode=args.image_url_mode,
        embed_images=args.embed_images,
        crop_root=args.crop_root,
    )
    print(f'Prepared {len(tasks)} Label Studio tasks')

    if args.dry_run:
        return 0

    access_token = args.access_token
    if not access_token:
        if not args.refresh_token:
            print(
                'Error: provide --access-token or --refresh-token for Label Studio API calls',
                file=sys.stderr,
            )
            return 1
        access_token = refresh_access_token(args.label_studio_url, args.refresh_token)

    project_id = args.project_id
    if args.create_project:
        project_id = create_project(
            args.label_studio_url,
            access_token=access_token,
            title=args.project_title,
            label_config=build_label_config(),
        )
        print(f'Created Label Studio project: {project_id}')

    if project_id is None:
        print(
            'Error: provide --project-id or pass --create-project',
            file=sys.stderr,
        )
        return 1

    result = import_tasks(
        args.label_studio_url,
        access_token=access_token,
        project_id=project_id,
        tasks=tasks,
    )
    print(f'Imported tasks into project {project_id}: {result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
