from setuptools import find_packages, setup

project_name = "myproj"
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=project_name,
    packages=[project_name],
    package_dir={project_name: "src"},
    version="0.1.0",
    description="my desc",
    author="Yaron",
    author_email="author@example.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)
