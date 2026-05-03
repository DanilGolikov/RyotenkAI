"""Shared constants for providers (RunPod, SingleNode, etc.)."""

# HTTP methods (WPS226)
HTTP_GET = "GET"
HTTP_POST = "POST"

# Timeouts (seconds) - WPS432
TIMEOUT_REQUEST_DEFAULT = 30
TIMEOUT_REQUEST_SHORT = 60
TIMEOUT_REQUEST_LONG = 120
HTTP_STATUS_ERROR_THRESHOLD = 400
HTTP_STATUS_OK = 200

# Retry / backoff
RETRY_BACKOFF_FACTOR = 2.0
RETRY_TOTAL_ATTEMPTS = 3

# JSON/dict keys shared across >=2 provider modules (WPS226)
KEY_ID = "id"
KEY_NAME = "name"
KEY_QUERY = "query"
KEY_ERRORS = "errors"

SSH_PORT_DEFAULT = 22

SHA12_LEN = 12

# Encoding
ENCODING_UTF8 = "utf-8"

# Category label used in inference event loggers
CATEGORY_INFERENCE = "inference"
