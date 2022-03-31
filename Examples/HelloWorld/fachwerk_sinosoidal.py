
from fachwerk_lib import *
from RFEM.initModel import *

Model(new_model=True, model_name="fachwerk_sinosoidal")
# SetModelType(model_type=ModelType.E_MODEL_TYPE_2D_XZ_PLANE_STRESS)
SetModelType(model_type=ModelType.E_MODEL_TYPE_3D)
Material(1, "GL24h")
Section(1, "SQ_M1 50")
Section(2, "SQ_M1 20")

x = np.linspace(0, 2 * np.pi, 51)

coords = np.zeros((x.size, 3))
coords[:, 0] = x / (2 * np.pi) * 10
coords[:, 2] = 0.3 * np.sin(x)

num_fields = 10
Model.clientModel.service.begin_modification()
node_no = place_top_bot_members(coords, height=1)
place_beams_connection(node_no, num_fields, pattern="\\/|")

coords[:, 2] = coords[:, 2] - 4
node_no = place_top_bot_members(coords, height=1)
place_beams_connection(node_no, num_fields, pattern="/\\|")
# place_beams(node_no, num_fields, pattern="\\/|")
Model.clientModel.service.finish_modification()