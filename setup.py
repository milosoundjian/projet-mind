from setuptools import setup, find_packages

setup(
    name="pmind",          
    version="0.1.0",           
    description="Our PMIND package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)