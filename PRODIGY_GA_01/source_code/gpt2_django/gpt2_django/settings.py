"""
Django settings for the GPT-2 Prompt Generator project (PRODIGY_GA_01).

This project is intentionally minimal:
- No database is used or required (DATABASES is empty).
- Only the apps needed to render the single UI page are installed.
- It is meant to be run locally with `python manage.py runserver`.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: this key is fine for local/offline use only.
SECRET_KEY = "django-insecure-prodigy-ga-01-local-demo-key"

# DEBUG is on because this app is built to run on localhost only.
DEBUG = True

ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# Only what's needed to serve templates and static files for the UI.
INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "generator",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

ROOT_URLCONF = "gpt2_django.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "gpt2_django.wsgi.application"

# No database is needed for this app.
DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
