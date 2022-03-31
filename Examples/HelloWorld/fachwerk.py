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


def member_from_node_numbers(nodes: np.ndarray, section_no):
    """generates members from nodes array

    :param nodes: _description_
    :type nodes: np.ndarray
    :param section_no: _description_
    :type section_no: _type_
    :return: _description_
    :rtype: _type_
    """

    node_tup = convert_node_array_to_tuple(nodes)
    member_numbers = np.zeros(nodes.shape[0]).astype(int)
    for i in range(node_tup.shape[0]):
        member_numbers[i] = int(
            FirstFreeIdNumber(memType=ObjectTypes.E_OBJECT_TYPE_MEMBER)
        )
        Member(
            member_numbers[i],
            start_node_no=node_tup[i, 0],
            end_node_no=node_tup[i, 1],
            start_section_no=section_no,
            end_section_no=section_no,
        )
    return member_numbers


def place_nodes(coords: np.ndarray):
    """places nodes with a given numpy array

    :param coords: first col: x |second col: y | third col: z
    :type coords: np.ndarray
    :return: array of node numbers
    :rtype: np.ndarray
    """
    node_numbers = np.zeros(coords.shape[0]).astype(int)
    for i in range(coords.shape[0]):
        node_numbers[i] = int(FirstFreeIdNumber(memType=ObjectTypes.E_OBJECT_TYPE_NODE))
        Node(node_numbers[i], coords[i, 0], coords[i, 1], coords[i, 2])
    return node_numbers


def convert_node_array_to_tuple(node_numbers: np.ndarray):
    """converts a array of nodes to node tuple pairs for members

    :param node_numbers: array of nodes
    :type node_numbers: np.ndarray
    :return: returns tuple pairs
    :rtype: np.ndarray
    """

    node_tup = np.zeros((node_numbers.size - 1, 2)).astype(int)
    node_tup[:, 0] = node_numbers[:-1]
    node_tup[:, 1] = node_numbers[1:]
    return node_tup


def place_top_bot_members(coords: np.ndarray, height: float, section_no: int = 1):
    node_no_top = place_nodes(coords)
    member_from_node_numbers(node_no_top, section_no=section_no)
    height_offset = np.zeros(coords.shape)
    height_offset[:, 2] = height_offset[:, 2] + height
    node_no_bot = place_nodes(coords + height_offset)
    member_from_node_numbers(node_no_bot, section_no=section_no)
    node_no = np.zeros((node_no_top.size, 2)).astype(int)
    node_no[:, 0] = node_no_top
    node_no[:, 1] = node_no_bot
    return node_no


def extract_truss_nodes(node_no: np.ndarray, num_fields: int):
    vertical_indices = np.arange(
        0, node_no.shape[0], int(node_no.shape[0] / num_fields), dtype=int
    )
    return vertical_indices


def inject_node_with_offset(node_1, node_2, distance):
    node1 = Model.clientModel.service.get_node(node_1)
    node2 = Model.clientModel.service.get_node(node_2)
    # df = load_dataframe(Model.clientModel)
    node1_coords = np.array(
        [
            node1.global_coordinates.x,
            node1.global_coordinates.y,
            node1.global_coordinates.z,
        ]
    )
    node2_coords = np.array(
        [
            node2.global_coordinates.x,
            node2.global_coordinates.y,
            node2.global_coordinates.z,
        ]
    )
    vec = np.abs(node2_coords - node1_coords)
    vec_norm = vec / np.linalg.norm(vec)
    coords = (node1_coords + vec_norm * distance).reshape(-1, 3)
    node_num = place_nodes(coords)
    return node_num


def place_verticals(node_no: np.ndarray, num_fields: int, section_no: int = 1):
    node_no_top = node_no[:, 0]
    node_no_bot = node_no[:, 1]
    vertical_indices = extract_truss_nodes(node_no=node_no, num_fields=num_fields)
    for i in vertical_indices:
        member_from_node_numbers(
            np.array([node_no_top[i], node_no_bot[i]]), section_no=section_no
        )
    member_from_node_numbers(
        np.array([node_no_top[-1], node_no_bot[-1]]), section_no=section_no
    )


def place_verticals_connection(
    node_no: np.ndarray, num_fields: int, section_no: int = 1, **cs_props
):
    h_top = cs_props.get("h_top", 0.1)
    h_bot = cs_props.get("h_bot", 0.1)
    node_no_top = node_no[:, 0]
    node_no_bot = node_no[:, 1]
    indices = extract_truss_nodes(node_no=node_no, num_fields=num_fields)
    for i in indices:
        node_num_top = inject_node_with_offset(node_no_top[i], node_no_bot[i], h_top)[0]
        node_num_bot = inject_node_with_offset(node_no_bot[i], node_no_top[i], -h_bot)[
            0
        ]
        member_from_node_numbers(
            np.array([node_no_top[i], node_num_top]), section_no=section_no
        )
        member_from_node_numbers(
            np.array([node_num_top, node_num_bot]), section_no=section_no
        )
        member_from_node_numbers(
            np.array([node_num_bot, node_no_bot[i]]), section_no=section_no
        )

    node_num_top = inject_node_with_offset(node_no_top[-1], node_no_bot[-1], h_top)[0]
    node_num_bot = inject_node_with_offset(node_no_bot[-1], node_no_top[-1], -h_bot)[0]
    member_from_node_numbers(
        np.array([node_no_top[-1], node_num_top]), section_no=section_no
    )
    member_from_node_numbers(
        np.array([node_num_top, node_num_bot]), section_no=section_no
    )
    member_from_node_numbers(
        np.array([node_num_bot, node_no_bot[-1]]), section_no=section_no
    )


def place_beams(
    node_no: np.ndarray,
    num_fields: int,
    pattern="\\",
    from_field=0,
    to_field=-1,
    section_no=1,
):
    node_no_top = node_no[:, 0]
    node_no_bot = node_no[:, 1]
    indices = extract_truss_nodes(node_no=node_no, num_fields=num_fields)
    if pattern.replace("|", "") == "\\":
        for i in range(from_field, indices.size + to_field):
            member_from_node_numbers(
                np.array([node_no_top[indices[i]], node_no_bot[indices[i + 1]]]),
                section_no=section_no,
            )
    elif pattern.replace("|", "") == "/":
        for i in range(from_field, indices.size + to_field):
            member_from_node_numbers(
                np.array([node_no_bot[indices[i]], node_no_top[indices[i + 1]]]),
                section_no=section_no,
            )
    elif pattern.replace("|", "") == "/\\":
        for i in range(from_field, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_bot[indices[i]], node_no_top[indices[i + 1]]]),
                section_no=section_no,
            )
        for i in range(from_field + 1, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_top[indices[i]], node_no_bot[indices[i + 1]]]),
                section_no=section_no,
            )
    elif pattern.replace("|", "") == "\\/":
        for i in range(from_field + 1, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_bot[indices[i]], node_no_top[indices[i + 1]]]),
                section_no=section_no,
            )
        for i in range(from_field, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_top[indices[i]], node_no_bot[indices[i + 1]]]),
                section_no=section_no,
            )
    if "|" in pattern:
        place_verticals(node_no, num_fields)


def place_beams_connection(
    node_no: np.ndarray,
    num_fields: int,
    pattern="\\",
    from_field=0,
    to_field=-1,
    section_no=1,
    **cs_props
):
    h_top = cs_props.get("h_top", 0.1)
    h_bot = cs_props.get("h_bot", 0.1)
    node_no_top = node_no[:, 0]
    node_no_bot = node_no[:, 1]
    indices = extract_truss_nodes(node_no=node_no, num_fields=num_fields)
    if pattern.replace("|", "") == "\\":
        for i in range(from_field, indices.size + to_field):
            node_num_top = inject_node_with_offset(
                node_no_top[indices[i]], node_no_bot[indices[i + 1]], h_top
            )[0]
            node_num_bot = inject_node_with_offset(
                node_no_bot[indices[i + 1]], node_no_top[indices[i]], -h_bot
            )[0]
            member_from_node_numbers(
                np.array([node_no_top[indices[i]], node_num_top]),
                section_no=section_no,
            )
            member_from_node_numbers(
                np.array([node_num_top, node_num_bot]),
                section_no=section_no,
            )
            member_from_node_numbers(
                np.array([node_num_bot, node_no_bot[indices[i + 1]]]),
                section_no=section_no,
            )
    elif pattern.replace("|", "") == "/":
        for i in range(from_field, indices.size + to_field):
            member_from_node_numbers(
                np.array([node_no_bot[indices[i]], node_no_top[indices[i + 1]]]),
                section_no=section_no,
            )
    elif pattern.replace("|", "") == "/\\":
        for i in range(from_field, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_bot[indices[i]], node_no_top[indices[i + 1]]]),
                section_no=section_no,
            )
        for i in range(from_field + 1, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_top[indices[i]], node_no_bot[indices[i + 1]]]),
                section_no=section_no,
            )
    elif pattern.replace("|", "") == "\\/":
        for i in range(from_field + 1, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_bot[indices[i]], node_no_top[indices[i + 1]]]),
                section_no=section_no,
            )
        for i in range(from_field, indices.size + to_field, 2):
            member_from_node_numbers(
                np.array([node_no_top[indices[i]], node_no_bot[indices[i + 1]]]),
                section_no=section_no,
            )
    if "|" in pattern:
        place_verticals_connection(node_no, num_fields, **cs_props)


Model(new_model=True, model_name="fachwerk")
SetModelType(model_type=ModelType.E_MODEL_TYPE_2D_XZ_PLANE_STRESS)
Material(1, "GL24h")
Section(1, "SQ_M1 20")

x = np.linspace(0, 2 * np.pi, 51)

coords = np.zeros((x.size, 3))
coords[:, 0] = x / (2 * np.pi) * 10
coords[:, 2] = 0.3 * np.sin(x)

num_fields = 10
Model.clientModel.service.begin_modification()
node_no = place_top_bot_members(coords, height=1)
place_beams_connection(node_no, num_fields, pattern="\\|")
Model.clientModel.service.finish_modification()

# inject_node_with_offset(2, 3, 0.1)
# inject_node_with_offset(1, 52, 0.4)
# inject_node_with_offset(2, 3, 0)
