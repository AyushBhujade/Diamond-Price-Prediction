from setuptools import find_packages,setup
from typing import List

hypen_e_dot="-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        
        requirements=file_obj.readlines()
        requirements=[req.replace("/n","") for req in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
        return requirements
        
setup(
    name="Diamond Price Prediction",
    version='0.0.1',
    author="Ayush Bhujade",
    author_email="ayushbhujade@gmail.com" ,
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()
)