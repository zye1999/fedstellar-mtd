import logging
import os

from fedstellar import __version__


def collect_env():
    logging.debug("\n======== Fedstellar platform ========")
    logging.debug("Fedstellar platform version: " + str(__version__))
    logging.debug("Execution path:" + str(os.path.abspath(__file__)))

    logging.debug("\n======== Running Environment ========")
    import platform

    logging.debug("OS: " + platform.platform())
    logging.debug("Hardware: " + platform.machine())

    import sys

    logging.debug("Python version: " + sys.version)

    try:
        import torch
        logging.debug("PyTorch version: " + torch.__version__)
    except ImportError:
        logging.debug("PyTorch is not installed properly")

    logging.debug("\n======== CPU Configuration ========")

    try:
        import psutil

        # Getting loadover15 minutes
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = (load15 / os.cpu_count()) * 100

        logging.debug("The CPU usage is : {:.0f}%".format(cpu_usage))
        logging.debug(
            "Available CPU Memory: {:.1f} G / {}G".format(
                psutil.virtual_memory().available / 1024 / 1024 / 1024,
                psutil.virtual_memory().total / 1024 / 1024 / 1024,
            )
        )
    except ImportError:
        logging.debug("\n")

    try:
        logging.debug("\n======== GPU Configuration ========")

        logging.debug("Checking GPU... [NOT IMPLEMENTED YET]")

        import torch

        torch_is_available = torch.cuda.is_available()
        logging.debug("torch_is_available = {}".format(torch_is_available))

        device_count = torch.cuda.device_count()
        logging.debug("device_count = {}".format(device_count))

        device_name = torch.cuda.get_device_name(0)
        logging.debug("device_name = {}".format(device_name))

    except ImportError:
        logging.debug("No GPU devices")
