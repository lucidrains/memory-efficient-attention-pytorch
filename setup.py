from setuptools import setup, find_packages

setup(
  name = 'memory-efficient-attention-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.4',
  license='MIT',
  description = 'Memory Efficient Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/memory-efficient-attention-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention-mechanism'
  ],
  install_requires=[
    'einops>=0.3,<0.4',
    'torch>=1.6'    
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
