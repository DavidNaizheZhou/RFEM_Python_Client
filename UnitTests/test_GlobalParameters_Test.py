#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append(".")

# Importing the relevant libraries
from os import name
from RFEM.enums import *
from RFEM.globalParameter import *
from RFEM.dataTypes import *
from RFEM.initModel import *
from RFEM.BasicObjects.section import *
from RFEM.BasicObjects.material import *

def test_global_parameters():

    clientModel.service.begin_modification('new')
    #not yet implemented in RFEM6 GM
    GlobalParameter.AddParameter(GlobalParameter,
                                 no= 1,
                                 name= 'Test_1',
                                 symbol= 'Test_1',
                                 unit_group= GlobalParameterUnitGroup.LENGTH,
                                 definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_FORMULA,
                                 definition_parameter= ['1+1'],
                                 comment= 'Comment_1')
    # issue with optimization type
    # GlobalParameter.AddParameter(GlobalParameter,
    #                              no= 2,
    #                              name= 'Test_2',
    #                              symbol= 'Test_2',
    #                              unit_group= GlobalParameterUnitGroup.LOADS_DENSITY,
    #                              definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION,
    #                              definition_parameter= [50, 0, 100, 4],
    #                              comment= 'Comment_2')

    # GlobalParameter.AddParameter(GlobalParameter,
    #                             no= 3,
    #                             name= 'Test_3',
    #                             symbol= 'Test_3',
    #                             unit_group= GlobalParameterUnitGroup.AREA,
    #                             definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION_ASCENDING,
    #                             definition_parameter= [50, 0, 100, 4],
    #                             comment= 'Comment_3')

    # GlobalParameter.AddParameter(GlobalParameter,
    #                             no= 4,
    #                             name= 'Test_4',
    #                             symbol= 'Test_4',
    #                             unit_group= GlobalParameterUnitGroup.MATERIAL_QUANTITY_INTEGER,
    #                             definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION_ASCENDING,
    #                             definition_parameter= [50, 0, 100, 4],
    #                             comment= 'Comment_4')

    GlobalParameter.AddParameter(GlobalParameter,
                                no= 5,
                                name= 'Test_5',
                                symbol= 'Test_5',
                                unit_group= GlobalParameterUnitGroup.DIMENSIONLESS,
                                definition_type= GlobalParameterDefinitionType.DEFINITION_TYPE_VALUE,
                                definition_parameter= [0.25],
                                comment= 'Comment_5')

    print('Ready!')

    clientModel.service.finish_modification()

def test_get_list_of_parameters_formula_allowed_for():

    clientModel.service.begin_modification('new')
    Material(2, "S550GD 1.0531")
    Section(4, "Cable 14.00", 2)
    GlobalParameter.get_list_of_parameters_formula_allowed_for("", ObjectTypes.E_OBJECT_TYPE_SECTION, 4)
    print('Ready!')
    clientModel.service.finish_modification()

def test_set_and_get_formula():

    clientModel.service.begin_modification('new')
    Material(2)
    Section(4, "RHSPOI 400/150/10/45", 2)
    GlobalParameter.set_formula("", ObjectTypes.E_OBJECT_TYPE_SECTION,4,"area_shear_y","0.1448/100")
    formula = GlobalParameter.get_formula("",ObjectTypes.E_OBJECT_TYPE_SECTION,4,"area_shear_y")
    print('Ready!')
    clientModel.service.finish_modification()
    assert formula == "0.1448/100"
