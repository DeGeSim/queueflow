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
        logging.basicConfig(
            filename=filename,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%y-%m-%d %H:%M",
        )
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    streamhandler = RichHandler(
        log_time_format="%y-%m-%d %H:%M", highlighter=NullHighlighter()
    )
    if print_bool:
        logger.addHandler(streamhandler)
    logging_redirect_tqdm(logger)
    install_mp_handler(logger)
