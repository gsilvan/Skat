from setuptools import setup

setup(
    name='Skat',
    version='0.1.0',
    packages=['skat', 'skat.agents', 'tests'],
    url='https://www.trollbox.info/',
    license='GPL',
    author='Silvan Gümüsdere',
    author_email='silvan@trollbox.info',
    description='Skat game environment for building automated computer agents',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Education',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ]
)
