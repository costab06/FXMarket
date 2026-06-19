"""Entry point: python -m fx_market"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def main() -> None:
    import uvicorn
    from .settings import Settings
    settings = Settings()
    uvicorn.run(
        "fx_market.server.app:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
