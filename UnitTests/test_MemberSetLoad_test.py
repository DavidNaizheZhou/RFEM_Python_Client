import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__),
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)
from RFEM.Loads.membersetload import MemberSetLoad
from RFEM.LoadCasesAndCombinations.loadCase import LoadCase
from RFEM.LoadCasesAndCombinations.staticAnalysisSettings import StaticAnalysisSettings
from RFEM.TypesForNodes.nodalSupport import NodalSupport
from RFEM.BasicObjects.memberSet import MemberSet
from RFEM.BasicObjects.member import Member
from RFEM.BasicObjects.node import Node
from RFEM.BasicObjects.section import Section
from RFEM.BasicObjects.material import Material
from RFEM.initModel import Model, Calculate_all
from RFEM.enums import *

if Model.clientModel is None:
    Model()

def test_member_set_load():

    Model.clientModel.service.delete_all()
    Model.clientModel.service.begin_modification()

    # Create Material
    Material(1, 'S235')

    # Create Section
    Section(1, 'IPE 300')
    Section(2, 'CHS 100x4')

    # Create Nodes
    Node(1, 0.0, 0.0, 0.0)
    Node(2, 2, 0.0, 0.0)
    Node(3, 4, 0, 0)

    Node(4, 0, 5, 0)
    Node(5, 2, 5, 0)
    Node(6, 4, 5, 0)

    # Create Member
    Member(1, 1, 2, 0, 1)
    Member(2, 2, 3, 0, 1)
    Member(3, 4, 6, 0, 2)
    Member(4, 6, 5, 0, 2)

    # Create Member Set
    MemberSet(1, '1 2', SetType.SET_TYPE_CONTINUOUS)
    MemberSet(2, '3 4', SetType.SET_TYPE_CONTINUOUS)

    # Create Nodal Supports
    NodalSupport(1, '1 3 4 6', NodalSupportType.FIXED)

    # Create Static Analysis Settings
    StaticAnalysisSettings(1, '1. Order', StaticAnalysisType.GEOMETRICALLY_LINEAR)

    # Create Load Case
    LoadCase(1, 'DEAD', [True, 0.0, 0.0, 1.0])

    ## Initial Member Set Load ##
    MemberSetLoad(1, 1, '1', LoadDirectionType.LOAD_DIRECTION_LOCAL_Z, 5000)

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.Force(0, 2, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[5000])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM with Eccentricity ##
    MemberSetLoad.Force(0, 3, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[5000], force_eccentricity=True, params={'eccentricity_y_at_start' : 0.01, 'eccentricity_z_at_start': 0.02})

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM_TOTAL ##
    MemberSetLoad.Force(0, 4, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM_TOTAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[5000])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_1 ##
    MemberSetLoad.Force(0, 5, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_1, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, 5000, 1.2])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_N ##
    MemberSetLoad.Force(0, 6, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_N, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 5000, 2, 1, 2])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2x2 ##
    MemberSetLoad.Force(0, 7, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2x2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, False, 5000, 1, 2, 3])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2x ##
    MemberSetLoad.Force(0, 8, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 5000, 6000, 1, 2])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_VARYING ##
    MemberSetLoad.Force(0, 9, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 4000], [2, 1, 5000]])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.Force(0, 10, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 4000, 8000, 1, 2])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.Force(0, 11, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 4000, 8000, 1, 2])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.Force(0, 12, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[4000, 8000, 12000])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.Force(0, 13, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 4000], [2, 1, 5000]])

    ## Force Type Member Set Load with LOAD_DISTRIBUTION_VARYING_IN_Z ##
    MemberSetLoad.Force(0, 14, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING_IN_Z, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 4000], [2, 1, 5000]])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.Moment(0, 15, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[5000])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_1 ##
    MemberSetLoad.Moment(0, 16, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_1, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, 5000, 1.2])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_N ##
    MemberSetLoad.Moment(0, 17, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_N, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 5000, 2, 1, 2])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2x2 ##
    MemberSetLoad.Moment(0, 18, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2x2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, False, 5000, 1, 2, 3])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2x ##
    MemberSetLoad.Moment(0, 19, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 5000, 6000, 1, 2])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_VARYING ##
    MemberSetLoad.Moment(0, 20, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 4000], [2, 1, 5000]])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.Moment(0, 21, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 4000, 8000, 1, 2])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.Moment(0, 22, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[False, False, 4000, 8000, 1, 2])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.Moment(0, 23, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[4000, 8000, 12000])

    ## Moment Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.Moment(0, 24, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 4000], [2, 1, 5000]])

    ## Mass Type Member Set Load ##
    MemberSetLoad.Mass(0, 25, 1, mass_components=[1000])

    ## Temperature Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.Temperature(0, 26, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[18, 2])

    ## Temperature Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.Temperature(0, 27, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, 18, 20, False, False, 1, 2])

    ## Temperature Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.Temperature(0, 28, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, 18, 20, False, False, 1, 2])

    ## Temperature Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.Temperature(0, 29, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[1, 2, 3, 4, 5, 6])

    ## Temperature Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.Temperature(0, 30, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 285, 289], [2, 1, 293, 297]])

    ## TemperatureChange Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.TemperatureChange(0, 31, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[18, 2])

    ## TemperatureChange Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.TemperatureChange(0, 32, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, 18, 20, False, False, 1, 2])

    ## TemperatureChange Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.TemperatureChange(0, 33, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, 18, 20, False, False, 1, 2])

    ## TemperatureChange Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.TemperatureChange(0, 34, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[1, 2, 3, 4, 5, 6])

    ## TemperatureChange Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.TemperatureChange(0, 35, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 285, 289], [2, 1, 293, 297]])

    ## AxialStrain Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.AxialStrain(0, 36, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_X, load_parameter=[0.005])

    ## AxialStrain Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.AxialStrain(0, 37, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_X, load_parameter=[12, 16, False, False, 1, 2])

    ## AxialStrain Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.AxialStrain(0, 38, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_X, load_parameter=[12, 16, False, False, 1, 2])

    ## AxialStrain Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.AxialStrain(0, 39, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_X, load_parameter=[1, 2, 3])

    ## AxialStrain Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.AxialStrain(0, 40, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_X, load_parameter=[[1, 1, 285, 289], [2, 1, 293, 297]])

    ## AxialDisplacement Type Member Set Load ##
    MemberSetLoad.AxialDisplacement(0, 41, 1, '1', MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_X, 0.05)

    ## Precamber Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.Precamber(0, 42, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[0.005])

    ## Precamber Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.Precamber(0, 43, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, False, False, 1, 2])

    ## Precamber Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.Precamber(0, 44, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, False, False, 1, 2])

    ## Precamber Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.Precamber(0, 45, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[1, 2, 3])

    ## Precamber Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.Precamber(0, 46, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 285], [2, 1, 293]])

    ## InitialPrestress Type Member Set Load ##
    MemberSetLoad.InitialPrestress(0, 47, 1, '1', MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_X, 50)

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.Displacement(0, 48, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_1 ##
    MemberSetLoad.Displacement(0, 49, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_1, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, False, 1])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_N ##
    MemberSetLoad.Displacement(0, 50, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_N, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, False, False, 1, 2])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2x2 ##
    MemberSetLoad.Displacement(0, 51, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2x2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, False, False, False, 1, 2, 3])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2 ##
    MemberSetLoad.Displacement(0, 52, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, 0.6, False, False, 1, 2])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_VARYING ##
    MemberSetLoad.Displacement(0, 53, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [[0.001, 1, 1], [0.002, 2, 1]])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.Displacement(0, 54, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, False, False, 1, 2])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.Displacement(0, 55, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, False, False, 1, 2])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.Displacement(0, 56, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[1, 2, 3])

    ## Displacement Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.Displacement(0, 57, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 285], [2, 1, 293]])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_UNIFORM ##
    MemberSetLoad.Rotation(0, 58, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_UNIFORM, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_1 ##
    MemberSetLoad.Rotation(0, 59, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_1, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, False, 1])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_N ##
    MemberSetLoad.Rotation(0, 60, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_N, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, False, False, 1, 2])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2x2 ##
    MemberSetLoad.Rotation(0, 61, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2x2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, False, False, False, 1, 2, 3])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_2 ##
    MemberSetLoad.Rotation(0, 62, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_2, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [0.5, 0.6, False, False, 1, 2])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_CONCENTRATED_VARYING ##
    MemberSetLoad.Rotation(0, 63, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_CONCENTRATED_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, [[1, 1, 285], [2, 1, 293]])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_TRAPEZOIDAL ##
    MemberSetLoad.Rotation(0, 64, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TRAPEZOIDAL, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, False, False, 1, 2])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_TAPERED ##
    MemberSetLoad.Rotation(0, 65, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_TAPERED, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[12, 16, False, False, 1, 2])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_PARABOLIC ##
    MemberSetLoad.Rotation(0, 66, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_PARABOLIC, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[1, 2, 3])

    ## Rotation Type Member Set Load with LOAD_DISTRIBUTION_VARYING ##
    MemberSetLoad.Rotation(0, 67, 1, '1', MemberSetLoadDistribution.LOAD_DISTRIBUTION_VARYING, MemberSetLoadDirection.LOAD_DIRECTION_LOCAL_Z, load_parameter=[[1, 1, 285], [2, 1, 293]])

    ## Pipe Content Full Type Member Set Load ##
    MemberSetLoad.PipeContentFull(0, 68, 1, '2', specific_weight=5000)

    ## Pipe Content Partial Type Member Set Load ##
    MemberSetLoad.PipeContentPartial(0, 69, 1, '2', specific_weight=2000, filling_height=0.1)

    ## Pipe Internal Pressure Type Member Set Load ##
    MemberSetLoad.PipeInternalPressure(0, 70, 1, '2', 2000)

    ## Pipe Rotary Motion Type Member Set Load ##
    MemberSetLoad.RotaryMotion(0, 71, 1, '2', 3.5, 5,
                               MemberSetLoadAxisDefinitionType.AXIS_DEFINITION_TWO_POINTS,
                               MemberLoadAxisDefinitionAxisOrientation.AXIS_NEGATIVE,
                               MemberSetLoadAxisDefinition.AXIS_Y, [10,11,12], [0,5,6])

    #Calculate_all() # Don't use in unit tests. See template for more info.

    Model.clientModel.service.finish_modification()
