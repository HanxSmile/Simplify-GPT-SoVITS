"""Setup file."""
from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


version = {}
with open("gpt_sovits/__version__.py") as version_file:
    exec(version_file.read(), version)

setup(
    name="gpt_sovits",
    version=version["__version__"],
    description="Python SDK for GPT-SoVITS",
    long_description=readme(),
    long_description_content_type='text/markdown',
    author="hanxiao",
    author_email="hanx_smile@icloud.com",
    packages=find_packages(),
    package_data={
        "gpt_sovits": [
            "text/chinese/opencpop-strict.txt",
            "text/chinese/g2pw/polyphonic.pickle",
            "text/chinese/g2pw/polyphonic.rep",
            "text/chinese/g2pw/polyphonic-fix.rep",
        ]
    },
    include_package_data=True,
    install_requires=fetch_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
