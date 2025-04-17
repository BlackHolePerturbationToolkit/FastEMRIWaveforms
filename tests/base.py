"""Definition of baseclass for FEW tests"""

import abc
import gc
import inspect
import logging
import typing as t
import unittest

import wrapt

from few.cutils import Backend
from few.utils.baseclasses import ParallelModuleBase
from few.utils.globals import get_config, get_first_backend, get_logger


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
        _ = get_config()  # Forge config to be finalized
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


@wrapt.decorator
def high_memory(wrapped, instance, args, kwargs):
    if instance is None:
        if inspect.isclass(wrapped):
            # Decorator was applied to a class.
            get_logger().warning("TestSuite '%s' uses large memory", wrapped.__name__)
            return wrapped(*args, **kwargs)
        else:
            # Decorator was applied to a function or staticmethod.
            get_logger().warning("Test '%s' uses large memory", wrapped.__name__)
            return wrapped(*args, **kwargs)
    else:
        if inspect.isclass(instance):
            # Decorator was applied to a classmethod.
            raise NotImplementedError(
                "high_memory decorator should not be applied to classmethod."
            )
        else:
            # Decorator was applied to an instancemethod.
            class_name = type(instance).__name__
            test_name = wrapped.__name__
            get_logger().warning(
                "Test %s of class %s uses large memory", test_name, class_name
            )
            return wrapped(*args, **kwargs)


class need_files:
    _files: t.Sequence[str]

    def __init__(self, *files: str):
        self._files = files

    def apply_skip(self, wrapped, instance, reason):
        if instance is None and inspect.isclass(wrapped):
            # decorator applied on class
            for item in inspect.getmembers(wrapped):
                if inspect.isfunction(item[1]) and item[0].startswith("test_"):
                    setattr(wrapped, item[0], need_files(*self._files)(item[1]))
            return wrapped

        if instance is not None and not inspect.isclass(instance):
            # decorator applied on instance method
            def skipped_wrapped(*args, **kwargs):
                raise unittest.SkipTest(reason)

            return skipped_wrapped

        raise NotImplementedError

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        from few import get_file_manager

        file_manager = get_file_manager()
        for file in self._files:
            if file_manager.try_get_file(file) is None:
                wrapped = self.apply_skip(
                    wrapped, instance, f"File '{file}' is not available"
                )
                break

        return wrapped(*args, **kwargs)


@wrapt.decorator
def no_file(wrapped, instance, args, kwargs):
    """No-op decorator to indicate explicitely that test does not need file"""
    wrapped(*args, **kwargs)
