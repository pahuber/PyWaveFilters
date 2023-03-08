from abc import ABC, abstractmethod


class OpticalElement(ABC):
    def __init__(self):
        description = 'Optical element.'
        super().__init__()

    @abstractmethod
    def apply(self):
        pass