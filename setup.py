"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='libhandy',
    version='0.1',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    packages=find_packages(),
    package_data={
        # 'beeline_robot': ["dates/*", "icons/*"]
    },
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy',
    ],
    # extras_require={
    #     'kaz_download':['requests==2.13.0','beautifulsoup4==4.5.1'],
    #     'bercut_xls_parser':['xlrd'],
    #     'beeline_robot':['pyautogui'],
    #     'beeline_report':[
    #         'openpyxl',
    #         'numpy',
    #         'pandas',
    #         'sqlalchemy',
    #         'psycopg2'
    #     ],
    # },
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'libhandy=libhandy.core:main',
        ],
    },
)
