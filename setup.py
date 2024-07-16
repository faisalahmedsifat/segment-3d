from setuptools import setup, find_packages

setup(
    name='your_project_name',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'PyQt5',
        'vtk',
        'opencv-python',
        'torch',
        'segment-anything',
        'numpy',
        'matplotlib',
        'pillow',
        'stl',
        'trimesh',
        'open3d'
    ],
    entry_points={
        'console_scripts': [
            'your_project_name=src.main:main',
        ],
    },
)
