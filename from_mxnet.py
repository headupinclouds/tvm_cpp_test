"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy mxnet models with NNVM.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import nnvm
import tvm
import numpy as np

import argparse

target_table = [
    'llvm',
    'llvm -target=aarch64-linux-android',
    'cuda',
    'opengl',
    'opencl',
    'vulkan',
    'metal'
];

description="""
TVM from_mxnet tutorial with cross platform modifications\n
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=description
)

parser.add_argument(
    '--target',
    choices=[x for x in target_table],
    help="Target device for inference step",
)

parser.add_argument(
    '--target-host',
    default=None,
    help="Host device (cross platform usage)",
)

args = parser.parse_args()

target = args.target
target_host = None if (args.target_host == "None") else args.target_host

# detect android cross compilation
is_android = True if ('android' in (target + str(target_host))) else False

print("is_android ", is_android)

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
from matplotlib import pyplot as plt
block = get_model('resnet18_v1', pretrained=True)
img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
download(synset_url, synset_name)
with open(synset_name) as f:
    synset = eval(f.read())
image = Image.open(img_name).resize((224, 224))
#plt.imshow(image)
#plt.show()

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
print('x', x.shape)

######################################################################t
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
sym, params = nnvm.frontend.from_mxnet(block)
# we want a probability so add a softmax operator
sym = nnvm.sym.softmax(sym)

######################################################################
# now compile the graph
import nnvm.compiler

print("target ", target)
print("target_host ", target_host)

shape_dict = {'data': x.shape}

#    with tvm.build_config(unroll_explicit=False):

with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params, target_host=target_host)

# Export library (prepare for remote save/load, proof of concept)
# https://docs.tvm.ai/api/python/module.html#tvm.module.Module.export_library

#####################
### serialization ###
#####################

params_bytes = nnvm.compiler.save_param_dict(params)
with open('from_mxnet.params', 'bw') as f:
  f.write(params_bytes)

# https://docs.tvm.ai/api/python/nnvm/graph.html#nnvm.graph.Graph.json
graph_json = graph.json()

with open('from_mxnet.json', 'w') as f:
  f.write(graph_json)

with open('cat.bin', 'wb') as f:
  f.write(x.astype(np.float32).tobytes())

#print("source: ", lib.get_source())  

if is_android:
    lib.export_library('from_mxnet.so', tvm.contrib.ndk.create_shared, options=[
        "-g",
        "-shared",
        "-fPIC",
        "-nostdlib++"
    ])

    # TODO: we could enable the android rpc server for device testing in python
    # skip inference step for cross compilation for now
    exit()
else:
    lib.export_library('from_mxnet.so')


    
######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime

if target=='llvm':
    ctx = tvm.cpu(0)
elif target=='cuda':
    ctx = tvm.gpu(0)
elif target=='opengl':
    ctx = tvm.opengl(0)
elif target=='opencl':
    ctx = tvm.cl(0)
elif target=='vulkan':
    ctx = tvm.vulkan(0)
elif target=='metal':
    ctx = tvm.metal(0)
else:
    raise ValueError('No supported context type for ' % target)

dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('data', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)
top1 = np.argmax(tvm_output.asnumpy()[0])
print('TVM prediction top-1:', top1, synset[top1])

######################################################################
# Use MXNet symbol with pretrained weights
# ----------------------------------------
# MXNet often use `arg_prams` and `aux_params` to store network parameters
# separately, here we show how to use these weights with existing API
def block2symbol(block):
    data = mx.sym.Variable('data')
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs
mx_sym, args, auxs = block2symbol(block)
# usually we would save/load it as checkpoint
mx.model.save_checkpoint('resnet18_v1', 0, mx_sym, args, auxs)
# there are 'resnet18_v1-0000.params' and 'resnet18_v1-symbol.json' on disk

######################################################################
# for a normal mxnet model, we start from here
mx_sym, args, auxs = mx.model.load_checkpoint('resnet18_v1', 0)
# now we use the same API to get NNVM compatible symbol
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
# repeat the same steps to run this model using TVM

