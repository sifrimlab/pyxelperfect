import setuptools
from pathlib import Path
import setuptools
import sys
min_version = (3, 8)

base_dir = Path(__file__).parent.resolve()
version_file = base_dir / "__version__.py"
readme_file = base_dir / "README.md"

with version_file.open() as f:
    exec(f.read())

with readme_file.open(encoding = "utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name = "pyxelperfect",
    version = __version__,
    author = "LMIB",
    author_email = "david.wouters@kuleuven.be",
    description = "Collection of python powered image-processing functions",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = "image-processing",
    url = "https://github.com/sifrimlab/pyxelperfect",
    project_urls = {
        "Bug Reports": "https://github.com/sifrimlab/pyxelperfect",
        "Source": "https://github.com/sifrimlab/pyxelperfect",
    },
    packages = setuptools.find_packages(),
    python_requires = '>={}'.format('.'.join(str(n) for n in min_version)),
    install_requires = [
        "numpy",
        "tqdm",
        "scikit-image",
        "opencv-python",
        "pandas",
        "aicspylibczi",
        "matplotlib",
        "Pillow",
        "tifffile",
        "tabulate",
        "nd2reader",
        "stardist",
        "csbdeep",
        "tensorflow",
        "SimpleITK",
        "cellpose",
    ],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
