"""
Setup script for nokta-ai package
"""

from setuptools import setup, find_packages

# For backwards compatibility
setup(
    packages=find_packages(),
    package_data={
        'nokta_ai': ['*.yaml', '*.yml'],
    }
)