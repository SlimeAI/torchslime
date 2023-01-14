from setuptools import setup, find_packages
from torchslime import __version__

README = 'README.md'

setup(
    name='torchslime',
    version=__version__,
    packages=find_packages(),
    include_package_data=False,
    entry_points={},
    install_requires=[],
    url='https://github.com/johncaged/TorchSlime',
    author='Zikang Liu',
    author_email='liuzikang0625@gmail.com',
    license='MIT License',
    description='TorchSlime is a PyTorch-based framework that helps you easily build your DL projects with pre-defined, highly extensible pipelines and other utils.',
    long_description=open(README, encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
