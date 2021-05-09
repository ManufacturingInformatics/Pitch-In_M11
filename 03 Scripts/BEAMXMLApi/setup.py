from setuptools import setup, find_packages

with open("ReadMe.md",'r') as f:
      ldescr= f.read()

setup(name='beamxmlapi',
      version='1.0.0',
      description='Functions to parse XML trace logs from BEAM machine',
      long_description = ldescr,
      author='David Miller',
      author_email='d.b.miller@sheffield.ac.uk',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
	  'Programming Language :: Python :: 3.7',
          'Environment :: Win32 (MS Windows)'],
      packages=['beamxmlread'],
      include_package_data=True,
      python_requires='>=3',
      install_requires=['numpy'],
      )
