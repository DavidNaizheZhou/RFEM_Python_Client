#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

# Importing the relevant libraries
from RFEM.enums import GlobalParameterUnitGroup, GlobalParameterDefinitionType
from RFEM.globalParameter import GlobalParameter
from RFEM.enums import ObjectTypes
from RFEM.initModel import Model, SetAddonStatus
from RFEM.BasicObjects.section import Section
from RFEM.BasicObjects.material import Material

if Model.clientModel is None:
    Model()

def test_global_parameters():

    Model.clientModel.service.delete_all()
    Model.clientModel.service.begin_modification()

    GlobalParameter(no= 1,
                    name= 'a',
                    symbol= 'a',
                    unit_group= GlobalParameterUnitGroup.LENGTH,
                    definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_VALUE,
                    definition_parameter= ['10'],
                    comment= 'param a')

    GlobalParameter(no= 2,
                    name= 'b',
                    symbol= 'b',
                    unit_group= GlobalParameterUnitGroup.LENGTH,
                    definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_FORMULA,
                    definition_parameter= ['a+1.5'],
                    comment= 'b')

    # TODO: bug 25058 is ToReview
    SetAddonStatus(Model.clientModel, 'cost_estimation_active')
    GlobalParameter(no= 3,
                     name= 'c',
                     symbol= 'c',
                     unit_group= GlobalParameterUnitGroup.LOADS_DENSITY,
                     definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION,
                     definition_parameter= [12, 10, 90, 40],
                     comment= 'c')

    GlobalParameter(no= 4,
                     name= 'D',
                     symbol= 'D',
                     unit_group= GlobalParameterUnitGroup.AREA,
                     definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION_ASCENDING,
                     definition_parameter= [50, 0, 100, 10],
                     comment= 'D')

    GlobalParameter(no= 5,
                    name= 'E',
                    symbol= 'E',
                    unit_group= GlobalParameterUnitGroup.MATERIAL_QUANTITY_INTEGER,
                    definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION_ASCENDING,
                    definition_parameter= [50, 0, 100, 10],
                    comment= 'E')

    GlobalParameter(no= 6,
                    name= 'f',
                    symbol= 'f',
                    unit_group= GlobalParameterUnitGroup.DIMENSIONLESS,
                    definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_VALUE,
                    definition_parameter= [0.25],
                    comment= 'f')

    Model.clientModel.service.finish_modification()

    gp = Model.clientModel.service.get_global_parameter(2)
    assert gp.formula == 'a+1.5'
    gp = Model.clientModel.service.get_global_parameter(3)
    assert gp.increment == 2
    gp = Model.clientModel.service.get_global_parameter(4)
    assert gp.increment == 10
    gp = Model.clientModel.service.get_global_parameter(5)
    assert gp.increment == 10
    gp = Model.clientModel.service.get_global_parameter(6)
    assert gp.value == 0.25

def test_get_list_of_parameters_formula_allowed_for():

    Model.clientModel.service.begin_modification()
    Material(2, "S550GD 1.0531")
    Section(4, "Cable 14.00", 2)
    glob_params = GlobalParameter.get_list_of_parameters_formula_allowed_for("", ObjectTypes.E_OBJECT_TYPE_SECTION, 4)

    Model.clientModel.service.finish_modification()

    assert len(glob_params) == 4
    assert glob_params[0] == 'rotation_angle'
    assert glob_params[1] == 'depth_temperature_load'
    assert glob_params[2] == 'width_temperature_load'

def test_set_and_get_formula():

    Model.clientModel.service.begin_modification()
    Material(2)
    Section(4, "RHSPOI 400/150/10/45", 2)
    GlobalParameter.set_formula("", ObjectTypes.E_OBJECT_TYPE_SECTION,4,"area_shear_y","0.1448/100")
    formula = GlobalParameter.get_formula("",ObjectTypes.E_OBJECT_TYPE_SECTION,4,"area_shear_y")

    Model.clientModel.service.finish_modification()

    assert formula.formula == '0.1448/100'
    assert formula.is_valid == True
    assert round(formula.calculated_value, 7) == 0.0014489
