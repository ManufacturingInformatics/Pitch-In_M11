from setuptools import setup, find_packages


setup(name='BEAMWriteToHDF5',
      version='1.0.0',
      description='Writing BEAM data to HDF5 file',
      author='David Miller',
      author_email='d.b.miller@sheffield.ac.uk',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          ],
      py_modules=['beamwritedatatohdf5'],
      python_requires='>=3',
      install_requires=['h5py','numpy'],
      )
