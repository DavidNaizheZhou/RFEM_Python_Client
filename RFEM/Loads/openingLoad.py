from RFEM.initModel import *
from RFEM.enums import *

class OpeningLoad():

    def __init__(self,
                 no: int = 1,
                 load_case_no: int = 1,
                 openings: str = '1',
                 load_distribution = OpeningLoadDistribution.LOAD_DISTRIBUTION_UNIFORM_TRAPEZOIDAL,
                 load_direction = OpeningLoadDirection.LOAD_DIRECTION_LOCAL_Z,
                 comment: str = '',
                 params: dict = {}):
        pass