# ====================================================
# ECHOSIS CUSTOM EXCEPTIONS
# ====================================================
#
# Collection of custom exceptions
#

class InsufficientAnnotatorsError(Exception):
    pass

class UnequalLabelsError(Exception):
    pass

class InvalidFileExtensionError(Exception):
    pass