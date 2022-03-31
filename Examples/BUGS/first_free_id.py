import os
import sys

baseName = os.path.basename(__file__)
dirName = os.path.dirname(__file__)
print("basename:    ", baseName)
print("dirname:     ", dirName)
sys.path.append(dirName + r"/../..")

import numpy as np

from RFEM.initModel import *
from RFEM.BasicObjects.node import Node

Model(new_model=True, model_name="BUG")
SetModelType(model_type=ModelType.E_MODEL_TYPE_2D_XZ_PLANE_STRESS)

x = np.linspace(0, 2, 10)
node_numbers = np.zeros(x.shape[0]).astype(int)

for i in range(x.shape[0]):
    node_numbers[i] = int(FirstFreeIdNumber(memType=ObjectTypes.E_OBJECT_TYPE_NODE))
    Node(node_numbers[i], x[i], 0, 0)
