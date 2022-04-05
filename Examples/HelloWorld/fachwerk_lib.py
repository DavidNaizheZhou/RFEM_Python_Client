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


def util_num_to_ndarray(nodes: np.ndarray, nums):
    arr = np.array([])
    if isinstance(nums, (int, float)):
        arr = np.array([nums] * nodes.shape[0]).flatten()
    elif isinstance(nums, list):
        arr = np.array(nums).flatten()
    elif isinstance(nums, np.ndarray):
        arr = np.array([nums]).flatten()
    return arr.astype(int)


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


def generate_members_from_tuples(nodes: np.ndarray, section_no):
    """generates members from node array tuples

    :param nodes: _description_
    :type nodes: np.ndarray
    :param section_no: _description_
    :type section_no: _type_
    :return: _description_
    :rtype: _type_
    """

    section_no = util_num_to_ndarray(nodes, section_no)
    if len(nodes.shape) == 1:
        node_tup = convert_node_array_to_tuple(nodes)
    elif len(nodes.shape) == 2:
        node_tup = nodes
    member_numbers = np.zeros(nodes.shape[0]).astype(int)
    for i in range(node_tup.shape[0]):
        member_numbers[i] = int(
            FirstFreeIdNumber(memType=ObjectTypes.E_OBJECT_TYPE_MEMBER)
        )
        Member(
            member_numbers[i],
            start_node_no=node_tup[i, 0],
            end_node_no=node_tup[i, 1],
            start_section_no=section_no[i],
            end_section_no=section_no[i],
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
    """places top bot with offset and a single array

    :param coords: _description_
    :type coords: np.ndarray
    :param height: _description_
    :type height: float
    :param section_no: _description_, defaults to 1
    :type section_no: int, optional
    :return: _description_
    :rtype: _type_
    """

    node_no_top = place_nodes(coords)
    generate_members_from_tuples(node_no_top, section_no=section_no)
    height_offset = np.zeros(coords.shape)
    height_offset[:, 2] = height_offset[:, 2] + height
    node_no_bot = place_nodes(coords + height_offset)
    generate_members_from_tuples(node_no_bot, section_no=section_no)
    node_no = np.zeros((node_no_top.size, 2)).astype(int)
    node_no[:, 0] = node_no_top
    node_no[:, 1] = node_no_bot
    return node_no

def place_top_bot_members_2(coord_bot: np.ndarray, coord_top: np.ndarray, section_no: int = 1):
    """places top bot with two arrays

    :param coords: _description_
    :type coords: np.ndarray
    :param height: _description_
    :type height: float
    :param section_no: _description_, defaults to 1
    :type section_no: int, optional
    :return: _description_
    :rtype: _type_
    """
    node_no_top = place_nodes(coord_top)
    generate_members_from_tuples(node_no_top, section_no=section_no)
    node_no_bot = place_nodes(coord_bot)
    generate_members_from_tuples(node_no_bot, section_no=section_no)
    node_no = np.zeros((node_no_top.size, 2)).astype(int)
    node_no[:, 0] = node_no_top
    node_no[:, 1] = node_no_bot
    return node_no


def extract_truss_nodes(node_no: np.ndarray, num_fields: int):
    vertical_indices = np.arange(
        0, node_no.shape[0], int(node_no.shape[0] / num_fields), dtype=int
    )
    return vertical_indices


def inject_node_with_offset(node_1, node_2, distance, reverse=False):
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

    # if reverse==True:
    vec = node2_coords - node1_coords
    # else:
    #     vec = node2_coords - node1_coords

    vec_norm = vec / np.linalg.norm(vec)
    coords = (node1_coords + vec_norm * distance).reshape(-1, 3)
    node_num = place_nodes(coords)
    return node_num


def place_verticals(node_no: np.ndarray, num_fields: int, section_no: int = 1):
    node_no_top = node_no[:, 0]
    node_no_bot = node_no[:, 1]
    vertical_indices = extract_truss_nodes(node_no=node_no, num_fields=num_fields)

    for i in vertical_indices:
        generate_members_from_tuples(
            np.array([node_no_top[i], node_no_bot[i]]), section_no=section_no
        )
    generate_members_from_tuples(
        np.array([node_no_top[-1], node_no_bot[-1]]), section_no=section_no
    )


def place_verticals_connection(
    node_no: np.ndarray,
    num_fields: int,
    section_no_vert=1,
    section_no_con=1,
    **cs_props
):
    h_top = cs_props.get("h_top", 0.1)
    h_bot = cs_props.get("h_bot", 0.1)
    node_no_top = node_no[:, 0]
    node_no_bot = node_no[:, 1]
    indices = extract_truss_nodes(node_no=node_no, num_fields=num_fields)
    for i in indices:
        node_num_top = inject_node_with_offset(node_no_top[i], node_no_bot[i], h_top)[0]
        node_num_bot = inject_node_with_offset(node_no_bot[i], node_no_top[i], h_bot)[0]
        generate_members_from_tuples(
            np.array([node_no_top[i], node_num_top]), section_no=section_no_con
        )
        generate_members_from_tuples(
            np.array([node_num_top, node_num_bot]), section_no=section_no_vert
        )
        generate_members_from_tuples(
            np.array([node_num_bot, node_no_bot[i]]), section_no=section_no_con
        )

    node_num_top = inject_node_with_offset(node_no_top[-1], node_no_bot[-1], h_top)[0]
    node_num_bot = inject_node_with_offset(node_no_bot[-1], node_no_top[-1], h_bot)[0]
    generate_members_from_tuples(
        np.array([node_no_top[-1], node_num_top]), section_no=section_no_con
    )
    generate_members_from_tuples(
        np.array([node_num_top, node_num_bot]), section_no=section_no_vert
    )
    generate_members_from_tuples(
        np.array([node_num_bot, node_no_bot[-1]]), section_no=section_no_con
    )


def place_beams(
    node_no: np.ndarray,
    num_fields: int,
    pattern="\\",
    from_field=0,
    to_field=-1,
    section_no_vert=1,
    section_no_diag=2,
):

    node_no_top = node_no[:, 0]
    node_no_bot = node_no[:, 1]
    indices = extract_truss_nodes(node_no=node_no, num_fields=num_fields)
    member_nodes = np.zeros((indices.shape[0] - 1, 2), dtype=int)
    section_no = util_num_to_ndarray(member_nodes[:, 0], section_no_diag)
    if pattern.replace("|", "") == "\\":
        member_nodes[:, 0] = node_no_top[indices[:-1]]
        member_nodes[:, 1] = node_no_bot[indices[1:]]

    elif pattern.replace("|", "") == "/":
        member_nodes[:, 0] = node_no_bot[indices[:-1]]
        member_nodes[:, 1] = node_no_top[indices[1:]]

    elif pattern.replace("|", "") == "/\\":
        nodes_diag1_start = node_no_bot[indices[:-1:2]]
        nodes_diag1_end = node_no_top[indices[1::2]]
        member_nodes[: nodes_diag1_start.size, 0] = nodes_diag1_start
        member_nodes[: nodes_diag1_start.size, 1] = nodes_diag1_end
        nodes_diag2_start = node_no_top[indices[1:-1:2]]
        nodes_diag2_end = node_no_bot[indices[2::2]]
        member_nodes[nodes_diag1_start.size :, 0] = nodes_diag2_start
        member_nodes[nodes_diag1_start.size :, 1] = nodes_diag2_end

    elif pattern.replace("|", "") == "\\/":
        nodes_diag1_start = node_no_bot[indices[from_field + 1 : to_field : 2]]
        nodes_diag1_end = node_no_top[indices[from_field + 2 :: 2]]
        member_nodes[: nodes_diag1_start.size, 0] = nodes_diag1_start
        member_nodes[: nodes_diag1_start.size, 1] = nodes_diag1_end
        nodes_diag2_start = node_no_top[indices[from_field:to_field:2]]
        nodes_diag2_end = node_no_bot[indices[from_field + 1 :: 2]]
        member_nodes[nodes_diag1_start.size :, 0] = nodes_diag2_start
        member_nodes[nodes_diag1_start.size :, 1] = nodes_diag2_end

    generate_members_from_tuples(
        member_nodes,
        section_no=section_no,
    )

    if "|" in pattern:
        member_nodes = np.zeros((node_no_top[indices].shape[0], 2)).astype(int)
        member_nodes[:, 0] = node_no_top[indices]
        member_nodes[:, 1] = node_no_bot[indices]
        section_no = util_num_to_ndarray(member_nodes[:, 0], section_no_diag)
        generate_members_from_tuples(
            member_nodes,
            section_no=section_no,
        )
        # place_verticals(node_no, num_fields)


def place_beams_connection(
    node_no: np.ndarray,
    num_fields: int,
    pattern="\\",
    from_field=0,
    to_field=-1,
    section_no_diag=1,
    section_no_vert=1,
    section_no_con=1,
    **cs_props
):
    """_summary_

    :param node_no: _description_
    :type node_no: np.ndarray
    :param num_fields: _description_
    :type num_fields: int
    :param pattern: _description_, defaults to "\"
    :type pattern: str, optional
    :param from_field: _description_, defaults to 0
    :type from_field: int, optional
    :param to_field: _description_, defaults to -1
    :type to_field: int, optional
    :param section_no_diag: _description_, defaults to 1
    :type section_no_diag: int, optional
    :param section_no_vert: _description_, defaults to 1
    :type section_no_vert: int, optional
    :param section_no_con: _description_, defaults to 1
    :type section_no_con: int, optional
    """

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
                node_no_bot[indices[i + 1]], node_no_top[indices[i]], h_bot
            )[0]
            generate_members_from_tuples(
                np.array([node_no_top[indices[i]], node_num_top]),
                section_no=section_no_con,
            )
            generate_members_from_tuples(
                np.array([node_num_top, node_num_bot]),
                section_no=section_no_diag,
            )
            generate_members_from_tuples(
                np.array([node_num_bot, node_no_bot[indices[i + 1]]]),
                section_no=section_no_con,
            )

    elif pattern.replace("|", "") == "/":
        for i in range(from_field, indices.size + to_field):
            node_num_top = inject_node_with_offset(
                node_no_top[indices[i + 1]], node_no_bot[indices[i]], h_top
            )[0]
            node_num_bot = inject_node_with_offset(
                node_no_bot[indices[i]], node_no_top[indices[i + 1]], h_bot
            )[0]
            generate_members_from_tuples(
                np.array([node_no_bot[indices[i]], node_num_bot]),
                section_no=section_no_con,
            )
            generate_members_from_tuples(
                np.array([node_num_bot, node_num_top]),
                section_no=section_no_diag,
            )
            generate_members_from_tuples(
                np.array([node_num_top, node_no_top[indices[i + 1]]]),
                section_no=section_no_con,
            )
            # generate_members_from_tuples(
            #     np.array([node_no_bot[indices[i]], node_no_top[indices[i + 1]]]),
            #     section_no=section_no,
            # )
    elif pattern.replace("|", "") == "/\\":
        for i in range(from_field, indices.size + to_field, 2):
            node_num_top = inject_node_with_offset(
                node_no_top[indices[i + 1]], node_no_bot[indices[i]], h_top
            )[0]
            node_num_bot = inject_node_with_offset(
                node_no_bot[indices[i]], node_no_top[indices[i + 1]], h_bot
            )[0]
            generate_members_from_tuples(
                np.array([node_no_bot[indices[i]], node_num_bot]),
                section_no=section_no_con,
            )
            generate_members_from_tuples(
                np.array([node_num_bot, node_num_top]),
                section_no=section_no_diag,
            )
            generate_members_from_tuples(
                np.array([node_num_top, node_no_top[indices[i + 1]]]),
                section_no=section_no_con,
            )
        for i in range(from_field + 1, indices.size + to_field, 2):
            node_num_top = inject_node_with_offset(
                node_no_top[indices[i]], node_no_bot[indices[i + 1]], h_top
            )[0]
            node_num_bot = inject_node_with_offset(
                node_no_bot[indices[i + 1]], node_no_top[indices[i]], h_bot
            )[0]
            generate_members_from_tuples(
                np.array([node_no_top[indices[i]], node_num_top]),
                section_no=section_no_con,
            )
            generate_members_from_tuples(
                np.array([node_num_top, node_num_bot]),
                section_no=section_no_diag,
            )
            generate_members_from_tuples(
                np.array([node_num_bot, node_no_bot[indices[i + 1]]]),
                section_no=section_no_con,
            )
    elif pattern.replace("|", "") == "\\/":
        for i in range(from_field + 1, indices.size + to_field, 2):
            node_num_top = inject_node_with_offset(
                node_no_top[indices[i + 1]], node_no_bot[indices[i]], h_top
            )[0]
            node_num_bot = inject_node_with_offset(
                node_no_bot[indices[i]], node_no_top[indices[i + 1]], h_bot
            )[0]
            generate_members_from_tuples(
                np.array([node_no_bot[indices[i]], node_num_bot]),
                section_no=section_no_con,
            )
            generate_members_from_tuples(
                np.array([node_num_bot, node_num_top]),
                section_no=section_no_diag,
            )
            generate_members_from_tuples(
                np.array([node_num_top, node_no_top[indices[i + 1]]]),
                section_no=section_no_con,
            )
        for i in range(from_field, indices.size + to_field, 2):
            node_num_top = inject_node_with_offset(
                node_no_top[indices[i]], node_no_bot[indices[i + 1]], h_top
            )[0]
            node_num_bot = inject_node_with_offset(
                node_no_bot[indices[i + 1]], node_no_top[indices[i]], h_bot
            )[0]
            generate_members_from_tuples(
                np.array([node_no_top[indices[i]], node_num_top]),
                section_no=section_no_con,
            )
            generate_members_from_tuples(
                np.array([node_num_top, node_num_bot]),
                section_no=section_no_diag,
            )
            generate_members_from_tuples(
                np.array([node_num_bot, node_no_bot[indices[i + 1]]]),
                section_no=section_no_con,
            )

    if "|" in pattern:
        place_verticals_connection(
            node_no,
            num_fields,
            section_no_vert=section_no_vert,
            section_no_con=section_no_con,
            **cs_props
        )


def fachwerk_sinosoidal():
    Model(new_model=True, model_name="fachwerk_sinosoidal")
    # SetModelType(model_type=ModelType.E_MODEL_TYPE_2D_XZ_PLANE_STRESS)
    SetModelType(model_type=ModelType.E_MODEL_TYPE_3D)
    Material(1, "GL24h")
    Material(2, "S235JRH")
    Section(1, "R_M1 50/50")
    Section(2, "R_M1 20/20")
    Section(3, "ROUND 5/H", material_no=2)

    section_OG = dict(Model.clientModel.service.get_section(1))
    section_UG = dict(Model.clientModel.service.get_section(1))
    height_dict = {
        "h_top": section_OG.get("depth_temperature_load") / 2,
        "h_bot": section_UG.get("depth_temperature_load") / 2,
    }

    x = np.linspace(0, 2 * np.pi, 51)

    coords = np.zeros((x.size, 3))
    coords[:, 0] = x / (2 * np.pi) * 10
    coords[:, 2] = 0.3 * np.sin(x)

    num_fields = 10
    Model.clientModel.service.begin_modification()
    node_no = place_top_bot_members(coords, height=1)
    place_beams_connection(
        node_no,
        num_fields,
        pattern="\\/|",
        section_no_vert=1,
        section_no_diag=2,
        section_no_con=3,
        **height_dict
    )

    coords = np.zeros((x.size, 3))
    coords[:, 0] = x / (2 * np.pi) * 10
    coords[:, 2] = 0.3 * np.cos(x) - 2

    node_no = place_top_bot_members(coords, height=1)
    place_beams_connection(
        node_no,
        num_fields,
        pattern="/\\|",
        section_no_vert=1,
        section_no_diag=2,
        section_no_con=3,
        **height_dict
    )
    # place_beams(node_no, num_fields, pattern="\\/|")
    Model.clientModel.service.finish_modification()


def fachwerk():
    Model(new_model=True, model_name="fachwerk")
    # SetModelType(model_type=ModelType.E_MODEL_TYPE_2D_XZ_PLANE_STRESS)
    SetModelType(model_type=ModelType.E_MODEL_TYPE_3D)
    Material(1, "GL24h")
    Material(2, "S235JRH")
    Section(1, "R_M1 50/50")
    Section(2, "R_M1 20/20")
    Section(3, "ROUND 5/H", material_no=2)

    section_OG = dict(Model.clientModel.service.get_section(1))
    section_UG = dict(Model.clientModel.service.get_section(1))
    height_dict = {
        "h_top": section_OG.get("depth_temperature_load") / 2,
        "h_bot": section_UG.get("depth_temperature_load") / 2,
    }

    x = np.linspace(0, 2 * np.pi, 51)

    coords = np.zeros((x.size, 3))
    coords[:, 0] = x / (2 * np.pi) * 10
    coords[:, 2] = 0

    num_fields = 10
    Model.clientModel.service.begin_modification()
    node_no = place_top_bot_members(coords, height=1)
    place_beams_connection(
        node_no,
        num_fields,
        pattern="\\/|",
        section_no_vert=1,
        section_no_diag=2,
        section_no_con=3,
        **height_dict
    )

    coords[:, 2] = coords[:, 2] - 2
    node_no = place_top_bot_members(coords, height=1)
    place_beams_connection(
        node_no,
        num_fields,
        pattern="/\\|",
        section_no_vert=1,
        section_no_diag=2,
        section_no_con=3,
        **height_dict
    )

    # place_beams(node_no, num_fields, pattern="\\/|")
    Model.clientModel.service.finish_modification()


def fachwerk_half_circle():
    Model(new_model=True, model_name="fachwerk_spiral")
    # SetModelType(model_type=ModelType.E_MODEL_TYPE_2D_XZ_PLANE_STRESS)
    SetModelType(model_type=ModelType.E_MODEL_TYPE_3D)
    Material(1, "GL24h")
    Material(2, "S235JRH")
    Section(1, "R_M1 50/50")
    Section(2, "R_M1 20/20")
    Section(3, "ROUND 5/H", material_no=2)

    section_OG = dict(Model.clientModel.service.get_section(1))
    section_UG = dict(Model.clientModel.service.get_section(1))
    height_dict = {
        "h_top": section_OG.get("depth_temperature_load") / 2,
        "h_bot": section_UG.get("depth_temperature_load") / 2,
    }

    periods = 1
    t = np.linspace(0, 2 * np.pi * periods, 60 * periods + 1)
    R = np.linspace(4, 7, t.size)
    R = 6
    a = 3

    coords_top = np.zeros((t.size, 3))
    coords_top[:, 0] = R * np.sin(t)  # x
    coords_top[:, 1] = R * np.cos(t)  # y
    coords_top[:, 2] = 0

    R = np.linspace(3, 6, t.size)
    coords_bot = np.zeros((t.size, 3))
    coords_bot[:, 0] = R * np.sin(t)  # x
    coords_bot[:, 1] = R * np.cos(t)  # y
    coords_bot[:, 2] = 2

    num_fields = 20 * periods
    Model.clientModel.service.begin_modification()
    node_no = place_top_bot_members_2(coord_bot=coords_bot, coord_top=coords_top)
    place_beams_connection(
        node_no,
        num_fields,
        pattern="/\\|",
        section_no_vert=1,
        section_no_diag=2,
        section_no_con=3,
        **height_dict
    )
    Model.clientModel.service.finish_modification()


def fachwerk_spiral():
    Model(new_model=True, model_name="fachwerk_spiral")
    # SetModelType(model_type=ModelType.E_MODEL_TYPE_2D_XZ_PLANE_STRESS)
    SetModelType(model_type=ModelType.E_MODEL_TYPE_3D)
    Material(1, "GL24h")
    Material(2, "S235JRH")
    Section(1, "R_M1 50/50")
    Section(2, "R_M1 20/20")
    Section(3, "ROUND 5/H", material_no=2)

    section_OG = dict(Model.clientModel.service.get_section(1))
    section_UG = dict(Model.clientModel.service.get_section(1))
    height_dict = {
        "h_top": section_OG.get("depth_temperature_load") / 2,
        "h_bot": section_UG.get("depth_temperature_load") / 2,
    }

    periods = 2
    t = np.linspace(0, 2 * np.pi * periods, 60 * periods + 1)
    R = np.linspace(4, 7, t.size)
    a = 3

    coords = np.zeros((t.size, 3))
    coords[:, 0] = R * np.sin(t)  # x
    coords[:, 1] = R * np.cos(t)  # y
    coords[:, 2] = a * t / (2 * np.pi)  # z

    num_fields = 20 * periods
    Model.clientModel.service.begin_modification()
    node_no = place_top_bot_members(coords, height=1)
    place_beams_connection(
        node_no,
        num_fields,
        pattern="/\\|",
        section_no_vert=1,
        section_no_diag=2,
        section_no_con=3,
        **height_dict
    )
    Model.clientModel.service.finish_modification()


if __name__ == "__main__":
    # fachwerk_sinosoidal()
    # fachwerk_spiral()
    # fachwerk()
    fachwerk_half_circle()
