if __name__ == "__main__":
    import unittest
    import pathlib

    current_directory = pathlib.Path(__file__).parent
    suite = unittest.TestLoader().discover(start_dir=current_directory)
    runner = unittest.TextTestRunner()
    runner.run(suite)
