import abc

class PsychoTest(abc.ABC):
    
    @abc.abstractmethod
    def test(self, model):
        ...
    
    @abc.abstractmethod
    def show_stimuli(self):
        ...

    @property
    @abc.abstractmethod
    def stimuli(self):
        ...