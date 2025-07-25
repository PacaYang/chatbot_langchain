import logging
from logging.handlers import RotatingFileHandler

# Logging 
# Create a rotating file handler
log_handler = RotatingFileHandler(
    'logs/chat_sessions.log',     # Log file name
    maxBytes=100_000_000,      # Rotate after 100 MB
    backupCount=5            # Keep up to 5 old log files
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Set formatter
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_handler.setFormatter(formatter)

# Apply to root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
