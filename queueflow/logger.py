import logging
from typing import Optional

from multiprocessing_logging import install_mp_handler
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def setup_logger(
    filename: Optional[str] = None, print_bool: bool = True, debug: bool = False
):
    if filename is not None:
        filehandler = logging.FileHandler(filename=filename, mode="w")
        logger.addHandler(filehandler)

    if print_bool:
        streamhandler = RichHandler(
            log_time_format="%y-%m-%d %H:%M", highlighter=NullHighlighter()
        )
        logger.addHandler(streamhandler)
