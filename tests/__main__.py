from unittest import TestLoader
from unittest.main import TestProgram


class TestFew(TestProgram):
    def createTests(self, from_discovery=False, Loader=None):
        import pathlib

        current_directory = pathlib.Path(__file__).parent
        self.test = TestLoader().discover(start_dir=current_directory)


if __name__ == "__main__":
    TestFew()
