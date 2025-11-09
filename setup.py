from setuptools import setup, find_packages

setup(
    name="brats-3d-segmentation",
    version="0.1.0",
    description="3D medical image segmentation for brain tumor detection using BraTS dataset",
    author="BraTS Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "monai[all]>=1.3.0",
        "nibabel>=5.1.0",
        "SimpleITK>=2.3.1",
        "captum>=0.6.0",
        "tensorboard>=2.14.0",
        "wandb>=0.15.8",
        "scikit-image>=0.21.0",
        "pandas>=2.0.3",
        "pyyaml>=6.0.1",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "connected-components-3d>=3.12.3",
        "scipy>=1.11.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)