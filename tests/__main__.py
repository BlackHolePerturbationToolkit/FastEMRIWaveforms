from unittest import TestLoader
from unittest.main import TestProgram


class TestFew(TestProgram):
    def createTests(self, from_discovery=False, Loader=None):
        import pathlib

        current_directory = pathlib.Path(__file__).parent
        self.test = TestLoader().discover(start_dir=current_directory)


if __name__ == "__main__":
    from few import get_file_manager, get_logger, get_config_setter

    get_config_setter().set_log_level("INFO")
    get_logger().info("Ensuring that all files required by tests are present.")
    get_file_manager().prefetch_files_by_tag("unittest")
    get_logger().info("Done... Now run the tests!")
    get_config_setter(reset=True)
    TestFew()
