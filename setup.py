from re import I
from setuptools import setup, find_packages

import os

setup(
    name = 'brisk',
    packages = find_packages(),
    install_requires = [
        'certifi==2021.10.8',
        'notebook==6.4.8',
        'numpy==1.22.3',
        'pandas==1.4.1',
        'python-dateutil==2.8.2',
        'pytz==2021.3',
        'six==1.16.0',
        'wincertstore==0.2'
    ],
    entry_points = {
        'console_scripts': [
            'hello_brisk=brisk.scripts.utils_scripts:hello',
            'brisk_load_imu=brisk.scripts.import:import_imu',
            'show_brisk_folders=brisk.scripts.utils_scripts:show_folders',
            'show_imu_config=brisk.scripts.utils_scripts:show_imu_config',
            'set_brisk_outdir=brisk.scripts.utils_scripts:set_out_dir',
        ]
    },
    package_data = {
        "": [os.path.join('config','*')],
    },
    version = "0.1",
    author = 'Simone Ranaldi',
    author_email='simone.ranaldi@uniroma3.it'
)