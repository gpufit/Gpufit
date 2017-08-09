"""
Discovers all tests and runs them. Assumes that initially the working directory is test.
"""

import sys
import unittest

if __name__ == '__main__':

    loader = unittest.defaultTestLoader

    tests = loader.discover('.')

    runner = unittest.TextTestRunner()

    results = runner.run(tests)

    # return number of failures
    sys.exit(len(results.failures))