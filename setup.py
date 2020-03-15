import setuptools
import re
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

package_name = "prsample"

version_file = os.path.join(package_name, "version.py")
verstr = "unknown"
try:
    verstrline = open(version_file, "rt").read()
except EnvironmentError:
    pass # Okay, there is no version file.
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        print ("unable to find version in %s" % (version_file,))
        raise RuntimeError("if %s.py exists, it is required to be well-formed" % (version_file,))

setuptools.setup(
    name=package_name, 
    version=verstr,
    author="Andrew Stanford-Jason",
    author_email="andrewstanfordjason@gmail.com",
    description="A pseudo-random data sampler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrewstanfordjason/prsample",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)