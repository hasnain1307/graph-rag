import os
import sys

from loguru import logger

from issm_api_common.config.settings import config


def setup_logger() -> logger:
    logger.remove()

    if not os.path.exists(config.log_file_path):
        os.makedirs(config.log_file_path)

    if config.log_to_file:
        # Set up file handler
        # Rotates the log file every day, or when the size is above 5 MB
        logger.add(
            config.log_file_path + "/file_{time:YYYY-MM-DD}.log",
            rotation="1 day",  # Rotate daily
            retention="10 days",  # Keep log files for 10 days
            level=config.log_level,  # Minimum level to log
            # Log message format
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            enqueue=True,  # Set to True for a more performant, but thread-safe, log writing
            backtrace=True,  # Set to True to get variable values in traceback
            diagnose=True,  # Set to True to get detailed traceback
        )

    if config.log_to_console:
        # Set up console handler
        logger.add(
            sys.stderr,
            level=config.log_level,  # Minimum level to log
            # Log message format
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            enqueue=True,  # Set to True for a more performant, but thread-safe, log writing
            backtrace=True,  # Set to True to get variable values in traceback
            diagnose=True,  # Set to True to get detailed traceback
        )

    return logger


logger = setup_logger()
