"""Console accounts are isolated from read-only passenger analytics."""
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / '.env.web', override=False)
SECRET_KEY = os.environ.get('WEB_SECRET_KEY', '')
if len(SECRET_KEY) < 32:
    raise ImproperlyConfigured('Create .env.web with python -m tools.setup_web_env')
DEBUG = os.environ.get('WEB_DEBUG', '0') == '1'
DEMO = os.environ.get('WEB_DEMO', '0') == '1'
ALLOWED_HOSTS = [value.strip() for value in os.environ.get('WEB_ALLOWED_HOSTS', 'localhost,127.0.0.1,[::1]').split(',') if value.strip()]
STATE_DIR = Path(os.environ.get('WEB_STATE_DIR', str(BASE_DIR / '.local' / 'web-console'))).resolve()
STATE_DIR.mkdir(parents=True, exist_ok=True)
INSTALLED_APPS = ['django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
                  'django.contrib.messages', 'django.contrib.staticfiles', 'web.portal', 'web.analytics']
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware', 'web.portal.middleware.PrivateResponsesMiddleware',
              'whitenoise.middleware.WhiteNoiseMiddleware',
              'django.contrib.sessions.middleware.SessionMiddleware', 'django.middleware.common.CommonMiddleware',
              'django.middleware.csrf.CsrfViewMiddleware', 'django.contrib.auth.middleware.AuthenticationMiddleware',
              'django.contrib.messages.middleware.MessageMiddleware', 'django.middleware.clickjacking.XFrameOptionsMiddleware']
ROOT_URLCONF = 'web.config.urls'
WSGI_APPLICATION = 'web.config.wsgi.application'
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates',
              'DIRS': [BASE_DIR / 'web' / 'templates'], 'APP_DIRS': True,
              'OPTIONS': {'context_processors': ['django.template.context_processors.request',
                          'django.contrib.auth.context_processors.auth', 'django.contrib.messages.context_processors.messages',
                          'web.portal.context_processors.console_context']}}]
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': STATE_DIR / 'accounts.sqlite3',
                         'OPTIONS': {'timeout': 15}}}
if DEMO:
    DATABASES['analytics'] = {'ENGINE': 'django.db.backends.sqlite3', 'NAME': STATE_DIR / 'demo-analytics.sqlite3'}
else:
    DATABASES['analytics'] = {
        'ENGINE': 'django.db.backends.postgresql', 'NAME': os.environ.get('WEB_ANALYTICS_DB_NAME', 'bus_analytics'),
        'HOST': os.environ.get('WEB_ANALYTICS_DB_HOST', '127.0.0.1'),
        'PORT': int(os.environ.get('WEB_ANALYTICS_DB_PORT', '5432')),
        'USER': os.environ.get('WEB_ANALYTICS_DB_USER', 'bus_dashboard'),
        'PASSWORD': os.environ.get('WEB_ANALYTICS_DB_PASS', ''), 'CONN_MAX_AGE': 0,
        'OPTIONS': {'connect_timeout': 3, 'options': '-c default_transaction_read_only=on -c statement_timeout=3000 -c lock_timeout=1000'},
    }
DATABASE_ROUTERS = ['web.analytics.router.AnalyticsRouter']
AUTH_USER_MODEL = 'portal.User'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 10}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]
LANGUAGE_CODE = 'ru'
TIME_ZONE = os.environ.get('WEB_TIMEZONE', 'Europe/Saratov')
USE_I18N = True
USE_TZ = True
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'web' / 'static']
STATIC_ROOT = STATE_DIR / 'static'
STORAGES = {'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'},
            'staticfiles': {'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage'}}
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'overview'
LOGOUT_REDIRECT_URL = 'login'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_AGE = 8 * 3600
CSRF_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = os.environ.get('WEB_SECURE_COOKIES', '0') == '1'
SECURE_SSL_REDIRECT = os.environ.get('WEB_REQUIRE_HTTPS', '0') == '1'
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
CSRF_TRUSTED_ORIGINS = [item for item in os.environ.get('WEB_CSRF_ORIGINS', '').split(',') if item]
# A trusted TLS reverse proxy can enable this explicitly. Loopback SSH needs no proxy headers.
if os.environ.get('WEB_TRUST_PROXY', '0') == '1':
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
FILE_UPLOAD_MAX_MEMORY_SIZE = DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024
CACHES = {'default': {'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
                      'LOCATION': str(STATE_DIR / 'cache')}}
LOGGING = {'version': 1, 'disable_existing_loggers': False,
           'handlers': {'console': {'class': 'logging.StreamHandler'}},
           'root': {'handlers': ['console'], 'level': 'INFO'}}
