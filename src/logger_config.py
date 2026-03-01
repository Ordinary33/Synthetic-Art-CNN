import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_PATH = PROJECT_ROOT / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str = "asl_classifier", level=logging.INFO):
    """Setup a logger with the specified name and logging level.

    Args:
        name (str, optional): _description_. Defaults to 'loan_classifier'.
        level (_type_, optional): _description_. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_filepath = LOGS_PATH / f"{name}.log"

    handler = logging.FileHandler(str(log_filepath))
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
