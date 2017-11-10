try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='apylisp',
    version='0.1',
    packages=['apylisp'],
    url='http://github.com/chrisrink10/apylisp',
    license='MIT License',
    author='Christopher Rink',
    author_email='chrisrink10@gmail.com',
    description='A Clojure-like lisp written for Python',
    install_requires=[
        'pyrsistent==0.14.0',
        'pytest==3.2.3'
    ],
)
