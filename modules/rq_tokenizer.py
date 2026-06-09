from enum import Enum

import gin


@gin.constants_from_enum
class RqTokenizerType(Enum):
    KMEANS = 1
    OPQ = 2
