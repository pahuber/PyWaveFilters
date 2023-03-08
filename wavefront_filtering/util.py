def get_kronecker_delta(index_a, index_b):
    '''
    Method to calculate the Kronecker delta.

            Parameters:
                    index_a (int): First index of Kronecker delta
                    index_b (int): Second index of Kronecker delta

            Returns:
                    (int): 0 or 1 as specified by the definition of the Kronecker delta
    '''

    if index_a == index_b:
        return 1
    else:
        return 0