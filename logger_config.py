import logging
from typing import Optional

DEFAULT_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "[{asctime}] {levelname}: {message}",
            "datefmt": "%H:%M:%S",
            "style": "{",
            "validate": True,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


def set_config(config: Optional[dict] = None) -> None:
    (
        logging.config.dictConfig(config)
        if config
        else logging.config.dictConfig(DEFAULT_CONFIG)
    )
