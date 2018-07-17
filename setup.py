import setuptools

setuptools.setup(
    name='keras-collections',
    version='1.0',
    description='A collection of deep learning models, and utility tookit written in keras and tensorflow',
    url='https://github.com/mingruimingrui/dataset-pipeline',
    author='Wang Ming Rui',
    author_email='mingruimingrui@hotmail.com',
    packages=[
        'keras_collections',
        'keras_collections.backend',
        'keras_collections.initializers',
        'keras_collections.layers',
        'keras_collections.losses',
        'keras_collections.models',
        'keras_collections.callbacks',
        'keras_collections.train',
        'keras_collections.inference',
        'keras_collections.utils'
    ]
)
