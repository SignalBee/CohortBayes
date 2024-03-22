from setuptools import setup
import os

with open(os.path.join("src/cohortbayes", "VERSION")) as version_file:
  version = version_file.read().strip()

setup(version=version,)
