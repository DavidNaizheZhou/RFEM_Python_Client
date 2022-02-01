from RFEM.initModel import Model, clearAtributes, ConvertToDlString
from RFEM.enums import SetType

class LineSet():
    def __init__(self,
                 no: int = 1,
                 lines_no: str = '33 36 39 42 45',
                 line_set_type = SetType.SET_TYPE_CONTINUOUS,
                 comment: str = '',
                 params: dict = {}):

        # Client model | Line Set
        clientObject = Model.clientModel.factory.create('ns0:line_set')

        # Clears object atributes | Sets all atributes to None
        clearAtributes(clientObject)

        # Line Set No.
        clientObject.no = no

        # Lines number
        clientObject.lines = ConvertToDlString(lines_no)

        # Line Set Type
        clientObject.set_type = line_set_type.name

        # Comment
        clientObject.comment = comment

        # Adding optional parameters via dictionary
        for key in params:
            clientObject[key] = params[key]

        # Add Line Set to client model
        Model.clientModel.service.set_line_set(clientObject)

    def ContinuousLines(self,
                 no: int = 1,
                 lines_no: str = '33 36 39 42 45',
                 comment: str = '',
                 params: dict = {}):

        # Client model | Line Set
        clientObject = Model.clientModel.factory.create('ns0:line_set')

        # Clears object atributes | Sets all atributes to None
        clearAtributes(clientObject)

        # Line Set No.
        clientObject.no = no

        # Lines number
        clientObject.lines = ConvertToDlString(lines_no)

        # Line Set Type
        clientObject.set_type = SetType.SET_TYPE_CONTINUOUS.name

        # Comment
        clientObject.comment = comment

        # Adding optional parameters via dictionary
        for key in params:
            clientObject[key] = params[key]

        # Add Line Set to client model
        Model.clientModel.service.set_line_set(clientObject)

    def GroupOfLines(self,
                 no: int = 1,
                 lines_no: str = '33 36 39 42 45',
                 comment: str = '',
                 params: dict = {}):

        # Client model | Line Set
        clientObject = Model.clientModel.factory.create('ns0:line_set')

        # Clears object atributes | Sets all atributes to None
        clearAtributes(clientObject)

        # Line Set No.
        clientObject.no = no

        # Lines number
        clientObject.lines = ConvertToDlString(lines_no)

        # Line Set Type
        clientObject.set_type = SetType.SET_TYPE_GROUP.name

        # Comment
        clientObject.comment = comment

        # Adding optional parameters via dictionary
        for key in params:
            clientObject[key] = params[key]

        # Add Line Set to client model
        Model.clientModel.service.set_line_set(clientObject)
