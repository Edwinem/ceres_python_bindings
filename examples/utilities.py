import os
import sys


def find_and_import_pyceres(folder_path = None):
    """Function which tries to import PyCeres given the multiple possible install options.

    """
    # Try importing it as if it was an installed library
    try:
        import PyCeres
        return
    except:
        pass

    # If it fails then try importing it from a given path
    if folder_path is not None:
        if not os.path.isdir(folder_path):
            raise ValueError("%s is not a folder, and therefore can't contain the PyCeres library.",folder_path)
        sys.path.insert(0, folder_path)
        try:
            import PyCeres
            return
        except:
            print("Will try to import from environment variable")
    # Try other importing options if the given path fails.
    pyceres_location="" # Folder where the PyCeres lib is created
    # try to import from environment variable
    if os.getenv('PYCERES_LOCATION'):
        pyceres_location=os.getenv('PYCERES_LOCATION')
    else:
        pyceres_location="../../build/lib" # If the environment variable is not set
    # then it will assume this directory. Only will work if built with Ceres and
    # through the normal mkdir build, cd build, cmake .. procedure
    sys.path.insert(0, pyceres_location)
    import PyCeres
    return


