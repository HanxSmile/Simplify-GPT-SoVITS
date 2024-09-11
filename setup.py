"""Setup file."""
import pip
from distutils.version import LooseVersion
from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


links = []
requires = []

if hasattr(pip, '__version__') and LooseVersion(str(pip.__version__)) >= LooseVersion('10.0.0'):
    # new versions of pip require a session
    from pip._internal import req, download

    requirements = req.parse_requirements('requirements.txt', session=download.PipSession())
elif hasattr(pip, '__version__') and LooseVersion(str(pip.__version__)) >= LooseVersion('7.0'):
    # new versions of pip require a session
    requirements = pip.req.parse_requirements('requirements.txt', session=pip.download.PipSession())
else:
    # old versions do not
    requirements = pip.req.parse_requirements('requirements.txt')

for item in requirements:
    # we want to handle package names and also repo urls
    if getattr(item, 'url', None):  # older pip has url
        links.append(str(item.url))
    if getattr(item, 'link', None):  # newer pip has link
        links.append(str(item.link))
    if item.req:
        requires.append(str(item.req))

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
    install_requires=requires,
    dependency_links=links,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
