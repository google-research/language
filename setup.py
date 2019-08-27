import os
from setuptools import setup, find_packages


def read(fname):
  return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
  name="language",
  version="0.0.1.dev",
  packages=find_packages(),
  description="Google AI Language.",
  long_description=read("README.md"),
  author="Google Inc.",
  url="https://github.com/google-research/language",
  license="MIT",
  install_requires=[
    "tensorflow-gpu==1.13.1",
  ],
  extras_require={
    "consistent-zero-shot-nmt": [
      "tensorflow-probability==0.6.0",
      "tensor2tensor==1.11.0",
    ],
 }
)
