import os
import sys

baseName = os.path.basename(__file__)
dirName = os.path.dirname(__file__)
print("basename:    ", baseName)
print("dirname:     ", dirName)
sys.path.append(dirName + r"/../..")

from RFEM.initModel import *
from RFEM.BasicObjects.node import Node

Model(new_model=True, model_name="extract")

# model modification
Model.clientModel.service.begin_modification()
Node(1, 0.0, 0.0, 0.0)
Node(2, 1.0, 0.0, 0.0)
Node(3, 2.0, 0.0, 0.0)
Model.clientModel.service.finish_modification()

# extract all nodes
numbers = ConvertStrToListOfInt(
    Model.clientModel.service.get_all_object_numbers(type="E_OBJECT_TYPE_NODE")
)
node_list = [Model.clientModel.service.get_node(int(n)) for n in numbers]
