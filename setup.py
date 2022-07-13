from setuptools import setup

setup(
    name='perceptualtests',
    version='0.1.0',    
    description='A module to test perceptual factors of deep learning models.',
    url='https://github.com/Jorgvt/PerceptualTests',
    author='Jorge Vila Tomás',
    author_email='jorge.vila-tomas@uv.es',
    license='BSD 2-clause',
    packages=['perceptualtests'],
    install_requires=[],
    include_package_data=True,
    package_data={'': ['data/*.mat']}
)
