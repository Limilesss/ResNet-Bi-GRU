from setuptools import setup, find_packages

setup(
    name='ResNet-Bi-GRU',
    version='1.0.0',
    description='A deep learning framework for time-series classification using ResNet + Bi-GRU',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorflow'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)