"""Library exceptions.
"""

class UnsupportedDataType(ValueError):
    """This exception is raised whenever a request is made to supply a column
    for a data type which is not supported.
    """
    pass
