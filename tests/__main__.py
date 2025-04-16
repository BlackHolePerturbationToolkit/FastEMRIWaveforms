from unittest import TestLoader
from unittest.main import TestProgram


class TestFew(TestProgram):
    def createTests(self, from_discovery=False, Loader=None):
        import pathlib

        current_directory = pathlib.Path(__file__).parent
        self.test = TestLoader().discover(start_dir=current_directory)


if __name__ == "__main__":
    import logging

    from few import get_config_setter, get_file_manager, get_logger

    if get_logger().getEffectiveLevel() > logging.INFO:
        get_config_setter(reset=True).set_log_level("INFO")

    get_logger().info("Ensuring that all files required by tests are present.")
    get_file_manager().prefetch_files_by_tag("unittest")
    get_logger().info("Done... Now run the tests!")
    get_config_setter(reset=True)
    TestFew()
