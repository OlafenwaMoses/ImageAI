from setuptools import setup,find_packages

setup(name="imageai",
      version='2.1.6',
      description='A python library built to empower developers to build applications and systems  with self-contained Computer Vision capabilities',
      url="https://github.com/OlafenwaMoses/ImageAI",
      author='Moses Olafenwa and John Olafenwa',
      author_email='guymodscientist@gmail.com',
      license='MIT',
      packages= find_packages(),
      install_requires=['numpy==1.19.3','scipy==1.4.1','pillow==8.1.1',"matplotlib==3.3.2", "h5py==2.10.0", "keras-resnet==0.2.0", "opencv-python", "keras==2.4.3"],
      zip_safe=False

      )