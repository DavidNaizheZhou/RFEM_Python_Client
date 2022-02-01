from RFEM.initModel import Model, clearAtributes
from RFEM.enums import GlobalParameterUnitGroup, GlobalParameterDefinitionType, ObjectTypes

class GlobalParameter():

    def __init__(self,
                 no: int = 1,
                 name: str = '',
                 symbol: str = '',
                 unit_group = GlobalParameterUnitGroup.LENGTH,
                 definition_type = GlobalParameterDefinitionType.DEFINITION_TYPE_VALUE,
                 definition_parameter = [],
                 comment: str = '',
                 params: dict = {}):
        '''
        for definition_type = GlobalParameterDefinitionType.DEFINITION_TYPE_FORMULA:
            definition_parameter = [formula]

        for definition_type = GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION:
            definition_parameter = [min, max, increment, steps]

        for definition_type = GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION_ASCENDING:
            definition_parameter = [min, max, increment, steps]

        for definition_type = GlobalParameterDefinitionType.DEFINITION_TYPE_OPTIMIZATION_DESCENDING:
            definition_parameter = [value, min, max, steps]

        for definition_type = GlobalParameterDefinitionType.DEFINITION_TYPE_VALUE:
            definition_parameter = [value]
        '''

        # Client model | Global Parameter
        clientObject = Model.clientModel.factory.create('ns0:global_parameter')

        # Clears object attributes | Sets all attributes to None
        clearAtributes(clientObject)

        # Global Parameter No.
        clientObject.no = no

        # Global Parameter Name
        clientObject.name = name

        # Symbol (HTML format)
        clientObject.symbol = symbol

        # Unit Group
        clientObject.unit_group = unit_group.name

        # Definition Type
        clientObject.definition_type = definition_type.name

        if definition_type.name == 'DEFINITION_TYPE_FORMULA':
            if len(definition_parameter) != 1:
                raise Exception('WARNING: The definition parameter needs to be of length 1. Kindly check list inputs for completeness and correctness.')
            clientObject.formula = definition_parameter[0]

        elif definition_type.name == 'DEFINITION_TYPE_OPTIMIZATION' or definition_type.name == 'DEFINITION_TYPE_OPTIMIZATION_ASCENDING' or definition_type.name == 'DEFINITION_TYPE_OPTIMIZATION_DESCENDING':
            if len(definition_parameter) != 4:
                raise Exception('WARNING: The definition parameter needs to be of length 4. Kindly check list inputs for completeness and correctness.')
            clientObject.value = definition_parameter[0]
            clientObject.min = definition_parameter[1]
            clientObject.max = definition_parameter[2]
            clientObject.steps = definition_parameter[3]

        elif definition_type.name == 'DEFINITION_TYPE_VALUE':
            if len(definition_parameter) != 1:
                raise Exception('WARNING: The definition parameter needs to be of length 1. Kindly check list inputs for completeness and correctness.')
            clientObject.value = definition_parameter[0]

        # Comment
        clientObject.comment = comment

        # Adding optional parameters via dictionary
        for key in params:
            clientObject[key] = params[key]

        # Add Global Parameter to client model
        Model.clientModel.service.set_global_parameter(clientObject)

    def get_list_of_parameters_formula_allowed_for(self,
                                                   object_type=ObjectTypes.E_OBJECT_TYPE_SECTION,
                                                   object_no: int = 1,
                                                   parent_no: int = 1):
        """
        Use this funtion to get all available properties for desired object type.

        Args:
            object_type (enum, required): Defaults to "E_OBJECT_TYPE_SECTION".
            object_no (int, required): Defaults to 1.
            parent_no (int, optional): Defaults to 1.
        Returns:
            allowed_params(list): list of all allowed parameters
        """
        # Client model | Object Location
        clientObject = Model.clientModel.factory.create('ns0:object_location')

        clientObject.type = object_type.name
        clientObject.no = object_no
        clientObject.parent_no = parent_no

        list_of_parameters = Model.clientModel.service.get_list_of_parameters_formula_allowed_for(clientObject)
        allowed_params = []
        for param in list_of_parameters.object_parameter_location:
            allowed_params.append(param.attribute)

        return allowed_params

    def get_formula(self,
                    object_type=ObjectTypes.E_OBJECT_TYPE_SECTION,
                    object_no: int = 1,
                    property: str = "area_shear_y",
                    parent_no: int = 1):
        """
        Returns formula for given object type, number and property.

        Args:
            object_type (enum, required): Defaults to "E_OBJECT_TYPE_GLOBAL_PARAMETER".
            object_no (int, required): Defaults to 1.
            property (string, required): Defaults to "area_shear_y".
            parent_no (int, optional): Defaults to 1.
        Returns:
            formula (str): formula for given type of object, numbrt, property and parent.
        """
        # Object Location
        object_location = Model.clientModel.factory.create('ns0:object_location')

        object_location.type = object_type.name
        object_location.no = object_no
        object_location.parent_no = parent_no

        # Object Parameter Location Type
        parameter_location = Model.clientModel.factory.create('ns0:object_parameter_location_type')
        parameter_location.attribute = property

        # Return Formula
        return Model.clientModel.service.get_formula(object_location, parameter_location)

    def set_formula(self,
                    object_type=ObjectTypes.E_OBJECT_TYPE_SECTION,
                    object_no: int = 1,
                    property: str = "area_shear_y",
                    formula: str = "1425/100",
                    parent_no: int = 1):
        """
        Set formula for given object type, number and property.

        Args:
            object_type (Enum, required): Defaults to "E_OBJECT_TYPE_GLOBAL_PARAMETER".
            object_no (int, required): Defaults to 1.
            property (string, required): Defaults to "area_shear_y".
            formula (string, required): Defaults to "beta/1000".
            parent_no (int, optional): Defaults to 1.
        """

        # Object Location
        object_location = Model.clientModel.factory.create('ns0:object_location')

        object_location.type = object_type.name
        object_location.no = object_no
        object_location.parent_no = parent_no

        # Object Parameter Location Type
        parameter_location = Model.clientModel.factory.create('ns0:object_parameter_location_type')
        parameter_location.attribute = property

        # Set Formula
        Model.clientModel.service.set_formula(object_location, parameter_location, formula)
