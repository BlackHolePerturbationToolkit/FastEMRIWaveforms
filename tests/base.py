"""Definition of baseclass for FEW tests"""

import abc
import gc
import logging
import typing as t
import unittest

from few.cutils import Backend
from few.utils.baseclasses import ParallelModuleBase
from few.utils.globals import get_first_backend, get_logger


class FewTest(unittest.TestCase, abc.ABC):
    """Baseclass for FEW tests with logger"""

    logger: logging.Logger

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Name of the test"""
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        cls.logger = get_logger()
        cls.logger.warning("Test '%s' is starting.", cls.name())
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.logger.warning("\n  Test '%s': done.", cls.name())
        del cls.logger
        gc.collect()
        super().tearDownClass()

    def tearDown(self):
        gc.collect()
        return super().tearDown()


class FewBackendTest(FewTest):
    """Base class for FEW tests with backend"""

    backend: Backend

    @classmethod
    @abc.abstractmethod
    def parallel_class(cls) -> t.Type[ParallelModuleBase]:
        """Class to be used as reference for the test backend selection"""
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.backend = get_first_backend(cls.parallel_class().supported_backends())
        cls.logger.warning(
            "  Test '%s': running with backend '%s'", cls.name(), cls.backend.name
        )

    @classmethod
    def tearDownClass(cls):
        del cls.backend
        super().tearDownClass()
