from abc import ABC, abstractmethod

from wavefront_filtering.wavefronts.wavefront import Wavefront


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
    def apply(self, wavefront: Wavefront):
        '''
        Abstract method to apply optical element.

                Parameters:
                        wavefront: Wavefront object
        '''
        pass
