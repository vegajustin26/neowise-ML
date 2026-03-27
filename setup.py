from setuptools import setup, find_packages

setup(name = "neowise-ML",
      package_dir = {"": "src"},
      packages = find_packages('src', include = ["neowise_ML"]),
      setup_requires=["numpy"],  # Just numpy here
    install_requires=["numpy", "tensorflow"],  # Add any of your other dependencies here
    )
