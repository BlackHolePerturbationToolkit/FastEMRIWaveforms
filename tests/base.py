"""Definition of baseclass for FEW tests"""

import abc
import dataclasses
import gc
import inspect
import logging
import typing as t
import unittest

import wrapt

from few.cutils import Backend
from few.utils import exceptions
from few.utils.baseclasses import ParallelModuleBase
from few.utils.globals import get_config, get_first_backend, get_logger


class FewTestDecorator:
    """Base class for few test decorators"""


class FewTestException(exceptions.FewException):
    """BaseClass for FEW Tests exceptions"""


@dataclasses.dataclass
class FlagTags:
    flag: str
    true_tag: str
    false_tag: str

    def from_kwargs(self, unary: bool = False, /, **kwargs: bool) -> str:
        flag = kwargs.get(self.flag) if self.flag in kwargs else True
        if unary:
            flag = not flag
        return self.true_tag if flag else self.false_tag


class tagged_test(FewTestDecorator):
    tags: t.Sequence[str]

    @staticmethod
    def exclusive_tags() -> t.Sequence[FlagTags]:
        return (
            FlagTags("slow", "slow", "fast"),
            FlagTags("high_memory", "high_memory", "low_memory"),
        )

    def _build_kwargs_tags(
        self, kwargs: dict[str, bool]
    ) -> t.Tuple[list[str], list[str]]:
        """Build list of activated and forbidden tags associated to kwargs"""
        flag_tags = self.exclusive_tags()
        activated_tags = [flag_tag.from_kwargs(**kwargs) for flag_tag in flag_tags]
        disabled_tags = [flag_tag.from_kwargs(True, **kwargs) for flag_tag in flag_tags]
        return activated_tags, disabled_tags

    def __init__(self, *tags: str, slow: bool = False, high_memory: bool = False):
        kwargs_tags, forbidden_tags = self._build_kwargs_tags(
            {"slow": slow, "high_memory": high_memory}
        )
        for forbidden_tag in forbidden_tags:
            if forbidden_tag in tags:
                raise FewTestException(
                    f"Cannot use tag '{forbidden_tag}' with tags '{kwargs_tags}'"
                )

        self.tags = set(list(tags) + kwargs_tags)

    def skip_if_disabled_tag(self, wrapped):
        def wrapped_with_skip(*args, **kwargs):
            disabled_tags = get_config().file_disabled_tags
            if disabled_tags is None:
                return wrapped(*args, **kwargs)

            for tag in self.tags:
                if tag in disabled_tags:
                    raise unittest.SkipTest(
                        f"Test skipped because tag '{tag}' is disabled."
                    )

            return wrapped(*args, **kwargs)

        return wrapped_with_skip

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        if instance is not None and not inspect.isclass(instance):
            # decorator applied on instance method
            return self.skip_if_disabled_tag(wrapped)(*args, **kwargs)
        raise NotImplementedError(
            "Decorator should only be applied to instance methods"
        )


class few_test_skipper:
    def skip_on_disabled_exception(self, wrapped):
        def skipped_wrapped(*args, **kwargs):
            try:
                wrapped(*args, **kwargs)
            except exceptions.FileManagerDisabledAccess as e:
                gc.collect()
                raise unittest.SkipTest(
                    f"Test uses file '{e.file_name}' which "
                    f"is disabled by tag '{e.disabled_tag}'."
                )

        return skipped_wrapped

    def apply_skip(self, wrapped, instance):
        if instance is not None and not inspect.isclass(instance):
            # decorator applied on instance method
            return self.skip_on_disabled_exception(wrapped)

        raise NotImplementedError(
            "Decorator should only be applied to instance methods, "
            f"was applied to {wrapped=}, {instance=}."
        )

    @wrapt.decorator
    def __call__(self, wrapped, instance, args, kwargs):
        wrapped = self.apply_skip(wrapped, instance)

        return wrapped(*args, **kwargs)


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

    @staticmethod
    def __apply_few_test_decorators(test):
        if not isinstance(test, tagged_test):
            test = tagged_test()(test)
        test = few_test_skipper()(test)
        return test

    def __init_subclass__(cls):
        """Apply skip-decorator on all test and setUp methods"""

        for item in inspect.getmembers(cls):
            if inspect.isfunction(item[1]) and (
                item[0].startswith("test_") or item[0] == "setUp"
            ):
                setattr(cls, item[0], cls.__apply_few_test_decorators(item[1]))
        super().__init_subclass__()


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
