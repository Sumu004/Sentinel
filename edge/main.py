"""Entry point: python -m edge.main [--no-preview]"""

from __future__ import annotations

import argparse
import logging

from edge.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sentinel edge pipeline.")
    parser.add_argument(
        "--no-preview", action="store_true", help="Don't open an OpenCV preview window."
    )
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run(show_preview=not args.no_preview)


if __name__ == "__main__":
    main()
