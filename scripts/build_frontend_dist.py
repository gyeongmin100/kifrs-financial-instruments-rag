"""Cloudflare Pages 업로드용 프론트엔드 번들을 생성한다.

사용법:
    python scripts/build_frontend_dist.py --api-base https://<space>.hf.space

결과: deploy/frontend_dist/ (index.html + static/*)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "src" / "accounting_rag" / "api" / "static"
DIST_DIR = PROJECT_ROOT / "deploy" / "frontend_dist"


def main() -> None:
    parser = argparse.ArgumentParser(description="프론트엔드 배포 번들 생성")
    parser.add_argument("--api-base", required=True, help="백엔드 주소 (예: https://xxx.hf.space)")
    args = parser.parse_args()
    api_base = args.api_base.rstrip("/")

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    (DIST_DIR / "static").mkdir(parents=True)

    shutil.copy2(STATIC_DIR / "index.html", DIST_DIR / "index.html")
    for name in ("styles.css", "app.js", "favicon.png"):
        shutil.copy2(STATIC_DIR / name, DIST_DIR / "static" / name)

    config = (
        "// 배포용 자동 생성 파일. scripts/build_frontend_dist.py가 만든다.\n"
        f'window.APP_CONFIG = {{ apiBase: "{api_base}" }};\n'
    )
    (DIST_DIR / "static" / "config.js").write_text(config, encoding="utf-8")
    print(f"생성 완료: {DIST_DIR}")
    print(f"apiBase: {api_base}")


if __name__ == "__main__":
    main()
