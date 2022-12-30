from setuptools import setup,find_packages

setup(name="imageai",
      version='3.0.3',
      description='A python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities',
      url="https://github.com/OlafenwaMoses/ImageAI",
      author='Moses Olafenwa',
      author_email='guymodscientist@gmail.com',
      license='MIT',
      packages= find_packages(exclude=["*imageai_tf_deprecated*"]),
      install_requires=[],
      include_package_data=True,
      zip_safe=False)