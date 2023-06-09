from abc import ABC, abstractmethod


class BaseOpticalElement(ABC):
    """
    Abstract class to represent optical elements.
    """

    def __init__(self):
        """
        Abstract contructor for optical element object.
        """
        self.description = 'Optical element.'
        super().__init__()

    @abstractmethod
    def apply(self):
        """
        Abstract method to apply optical element.

                Parameters:
                        wavefront: Wavefront object
        """
        pass
