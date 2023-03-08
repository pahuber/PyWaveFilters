from abc import ABC, abstractmethod


class OpticalElement(ABC):
    '''
    Abstract class to represent optical elements.
    '''

    def __init__(self):
        '''
        Abstract contructor for optical element object.
        '''
        description = 'Optical element.'
        super().__init__()

    @abstractmethod
    def apply(self):
        '''
        Abstract method to apply optical element.

                Parameters:
                        wavefront: Wavefront object
        '''
        pass
