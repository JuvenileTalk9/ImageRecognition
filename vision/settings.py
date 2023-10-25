import os
from pathlib import Path


# ベースディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent


# ロガー設定
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "production": {
            "format": "{asctime} [{levelname:8}] {module} line:{lineno:d} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{asctime} [{levelname:8}] {message}",
            "style": "{",
        },
    },
    "handlers": {
        "file": {
            "level": "DEBUG",
            "formatter": "production",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(BASE_DIR, "logs", "image_recognition.log"),
            "mode": "a",
            "maxBytes": 1024 * 1024 * 2,  # 2MB
            "backupCount": 3,  # 世代数
            "encoding": "utf-8",
        },
        "console": {
            "level": "DEBUG",
            "formatter": "simple",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "image_recognition": {
            "level": "DEBUG",
            "handlers": ["file", "console"],
            "propagate": False,
        },
    },
}
