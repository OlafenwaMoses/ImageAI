from setuptools import setup,find_packages

setup(name="imageai",
      version='2.2.0',
      description='A python library built to empower developers to build applications and systems with self-contained Computer Vision capabilities',
      url="https://github.com/OlafenwaMoses/ImageAI",
      author='Moses Olafenwa and John Olafenwa',
      author_email='guymodscientist@gmail.com',
      license='MIT',
      packages= find_packages(),
      install_requires=['tensorflow==2.9.1', 'keras==2.9.0', 'numpy==1.23.1', 'pillow==8.4.0', 'scipy==1.9.0', 'h5py==3.7.0', 'matplotlib==3.5.2', 'opencv-python==4.6.0.66', 'keras-resnet==0.2.0'],
      zip_safe=False
      )