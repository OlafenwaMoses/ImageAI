from os import path
from setuptools import setup

HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as fp:
    long_description = fp.read()

setup(
    name="imageai",
    version='2.1.3',
    description='A python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities',
    url="https://github.com/OlafenwaMoses/ImageAI",
    author='Moses Olafenwa and John Olafenwa',
    author_email='guymodscientist@gmail.com',
    license='MIT',
    packages=['imageai'],

    long_description=long_description,
    long_description_content_type='text/markdown',

    install_requires=['numpy','scipy','pillow',"matplotlib", "h5py", "keras", "opencv-python"],

    zip_safe=False,

    # Classifiers help users find your project by categorizing it.
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha ; 4 - Beta ; 5 - Production/Stable
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',

        # Specify the Python versions you support HERE. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)