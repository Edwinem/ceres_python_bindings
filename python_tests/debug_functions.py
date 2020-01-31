'''
Functions used to help with debugging
'''


def print_numpy_address(np_data):
    print(hex(np_data.__array_interface__['data'][0]))

def GetPIDAndPause():
    import os

    print(os.getpid())

    input("Enter to continue ...")