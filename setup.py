from setuptools import find_packages, setup
def get_requirements(file_path):

    #function to read requirements from a file
    hypen='-e .'
    with open(file_path) as f:
        requirements = f.readlines()
        if hypen in requirements:
            requirements.remove(hypen)
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
       
    return requirements
setup(
    name='mlproject',
    version='0.1.0',
    author='sakshi_porlekar',   
    author_email="sakshiporlekar27@gmail.com",
    description='A machine learning project setup',
    packages=find_packages(),
    install_requires=get_requirements('requirments.txt'),
    python_requires='>=3.7',
)               