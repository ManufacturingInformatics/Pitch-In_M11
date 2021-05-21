from setuptools import setup, find_packages

with open("ReadMe.md",'r') as f:
      ldescr= f.read()

setup(name='beamapi',
      version='1.0.0',
      description='BEAM PitchIn Project Modelling, Thermal Camera API and other functions',
      long_description = ldescr,
      author='David Miller',
      author_email='d.b.miller@sheffield.ac.uk',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Environment :: Win32 (MS Windows)'],
      packages=['LBAMExperimental','LBAMMaterials','LBAMModel','WriteDataToHPF5'],
      py_modules=['LBAMMaterials.LBAMMaterials',
                    'LBAMModel.LBAMModel',
                    'LBAMExperimental.LBAMExperimental',
                    'WriteDataToHPF5.beamwritedatatohdf5'],
      include_package_data=True,
      python_requires='>=3',
      install_requires=['numpy','scipy','h5py','opencv-python','scikit-image'],
      )
