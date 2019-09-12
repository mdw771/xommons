import setuptools

setuptools.setup(
    name='xommons',
    version='1.0.0',
    author='Ming Du',
    description='Quick calculation toolkit for x-ray imaging.',
    packages=setuptools.find_packages(exclude=['docs']),
    include_package_data=True,
    url='http://github.com/mdw771/xommons.git',
    keywords=['xommons'],
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers'
    ]
)
