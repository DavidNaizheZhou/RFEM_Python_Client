import os
import sys

baseName = os.path.basename(__file__)
dirName = os.path.dirname(__file__)
print("basename:    ", baseName)
print("dirname:     ", dirName)
sys.path.append(dirName + r"/../..")

import pandas as pd
import numpy as np

from RFEM.initModel import *
from RFEM.BasicObjects.node import Node
from RFEM.BasicObjects.material import Material
from RFEM.BasicObjects.section import Section
from RFEM.BasicObjects.member import Member
from RFEM.BasicObjects.line import Line


def load_object(clientModel, type="NODE"):
    numbers = ConvertStrToListOfInt(
        clientModel.service.get_all_object_numbers(
            type="E_OBJECT_TYPE_{}".format(type.upper())
        )
    )
    if type.upper() == "NODE":
        return [Model.clientModel.service.get_node(int(n)) for n in numbers]
    elif type.upper() == "MEMBER":
        return [Model.clientModel.service.get_member(int(n)) for n in numbers]
    elif type.upper() == "SECTION":
        return [Model.clientModel.service.get_section(int(n)) for n in numbers]
    else:
        # todo: other object types
        pass


def load_dataframe(clientModel, type="NODE"):
    object_list = load_object(clientModel, type)
    len_numbers = len(object_list)
    if len_numbers != 0:
        keys = dict(object_list[0]).keys()
        dataframe = pd.DataFrame(columns=keys, index=range(len_numbers), dtype=object)
        for i, node in enumerate(object_list):
            node_dict = dict(node)
            values = node_dict.values()
            dataframe.iloc[i] = np.array([value for value in values], dtype=object)
        return dataframe
    else:
        return None


def get_free_numbers(dataframe, num_range=range(1, 1000)):
    if isinstance(dataframe, pd.DataFrame):
        free_numbers = np.setxor1d(num_range, dataframe.no.values)
    else:
        free_numbers = num_range
    return free_numbers


def detail1(clientModel, coords, section_no_upper, section_no_lower):
    x, y, z = coords
    df_node = load_dataframe(clientModel, type="NODE")
    df_member = load_dataframe(clientModel, type="MEMBER")
    df_section = load_dataframe(clientModel, type="SECTION")

    free_node_numbers = get_free_numbers(df_node)
    free_member_numbers = get_free_numbers(df_member)

    dz = 0.1 / 2 - 0.05 / 2
    dx = 0.1 / 2 + 0.05

    Node(free_node_numbers[0], x - 1, y, z)
    Node(free_node_numbers[1], x - dx, y, z)
    Node(free_node_numbers[2], x - dx, y, z + dz)
    Node(free_node_numbers[3], x + dx, y, z + dz)
    Node(free_node_numbers[4], x + dx, y, z)
    Node(free_node_numbers[5], x + 1, y, z)

    indices = zip(np.arange(5), np.arange(5) + 1)
    for i, (index1, index2) in enumerate(indices):
        if (index1 == 2) & (index2 == 3):
            section_no = 2
            Member(
                free_member_numbers[i],
                start_node_no=free_node_numbers[index1],
                end_node_no=free_node_numbers[index2],
                start_section_no=section_no,
                end_section_no=section_no,
            )
        elif ((index1 == 1) & (index2 == 2)) or ((index1 == 3) & (index2 == 4)):
            Member(free_member_numbers[i]).Rigid(
                free_member_numbers[i],
                start_node_no=free_node_numbers[index1],
                end_node_no=free_node_numbers[index2],
            )
        else:
            section_no = 1
            Member(
                free_member_numbers[i],
                start_node_no=free_node_numbers[index1],
                end_node_no=free_node_numbers[index2],
                start_section_no=section_no,
                end_section_no=section_no,
            )

    dz = 0.1 / 2 - 0.05 / 2
    dy = 0.1 / 2 + 0.05

    Node(free_node_numbers[6], x, y - 1, z)
    Node(free_node_numbers[7], x, y - dy, z)
    Node(free_node_numbers[8], x, y - dy, z - dz)
    Node(free_node_numbers[9], x, y + dy, z - dz)
    Node(free_node_numbers[10], x, y + dy, z)
    Node(free_node_numbers[11], x, y + 1, z)

    indices = zip(np.arange(6, 6 + 5), np.arange(6, 6 + 5) + 1)
    for i, (index1, index2) in enumerate(indices):
        if (index1 == 8) & (index2 == 9):
            section_no = 2
            Member(
                free_member_numbers[i + 6],
                start_node_no=free_node_numbers[index1],
                end_node_no=free_node_numbers[index2],
                start_section_no=section_no,
                end_section_no=section_no,
            )
        elif ((index1 == 7) & (index2 == 8)) or ((index1 == 9) & (index2 == 10)):
            Member(free_member_numbers[i]).Rigid(
                free_member_numbers[i],
                start_node_no=free_node_numbers[index1],
                end_node_no=free_node_numbers[index2],
            )
        else:
            section_no = 1
            Member(
                free_member_numbers[i + 6],
                start_node_no=free_node_numbers[index1],
                end_node_no=free_node_numbers[index2],
                start_section_no=section_no,
                end_section_no=section_no,
            )

    # print(free_node_numbers[0:5])

    # clientModel.service.finish_modification()


Model(new_model=True, model_name="test1")
l = 5
Model.clientModel.service.begin_modification()
Material(1, "GL24h")
Section(1, "SQ_M1 100")
Section(2, "SQ_M1 50")

Node(1, 0.0, 0.0, 0.0)

detail1(Model.clientModel, (0, 0, 0), 1, 1)
detail1(Model.clientModel, (5, 0, 0), 1, 1)
# detail1(Model.clientModel, (10, 0, 0), 1, 1)
# detail1(Model.clientModel, (15, 0, 0), 1, 1)

Model.clientModel.service.finish_modification()

node_table = load_dataframe(Model.clientModel)
# print(node_table)
