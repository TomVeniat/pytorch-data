from setuptools import setup, find_packages


VERSION = '0.1'

requirements = [
    'torch',
    'tqdm',

]

setup(
    # Metadata
    name='pytorch-data',
    version=VERSION,
    author='Tom Veniat',
    author_email='veniat.tom@gmail.com',
    url='https://github.com/TomVeniat/pytorch-data',
    description='Additional datasets and transformers for pytorch',
    # long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(),

    zip_safe=True,
    install_requires=requirements,
)
