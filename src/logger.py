import logging
import os

from click import style


logging.addLevelName(logging.DEBUG, style(str(logging.getLevelName(logging.DEBUG)), fg="cyan"))
logging.addLevelName(logging.INFO, style(str(logging.getLevelName(logging.INFO)), fg="green"))
logging.addLevelName(logging.WARNING, style(str(logging.getLevelName(logging.WARNING)), fg="yellow"))
logging.addLevelName(logging.ERROR, style(str(logging.getLevelName(logging.ERROR)), fg="red"))
logging.addLevelName(logging.CRITICAL, style(str(logging.getLevelName(logging.CRITICAL)), fg="bright_red"))

logging.getLogger("aiocache").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
logging.getLogger("deepdiff.diff").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

log_format = "%(levelname)s: %(name)s:%(lineno)-3s %(message)s"
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
date_format = "%m-%d %H:%M:%S"
formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

handler = logging.StreamHandler()
handler.setLevel(log_level)
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.addHandler(handler)
