# setup.py

from setuptools import setup, find_packages

def load_requirements(filename: str):
    with open(filename) as f:
        return [x.strip() for x in f.readlines() if "-r" != x[0:2]]

setup(
    name='LauncherTemplate',
    version='0.3',
    packages=find_packages(),
    install_requires= load_requirements("requirements.txt"),
    author='NX',
    author_email='ningxiao@hust.edu.cn',
    description='LauncherTemplate',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Nx1021/LauncherTemplate.git',  # 替换为你的 Git 仓库地址
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
