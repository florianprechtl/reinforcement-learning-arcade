from abc import ABCMeta, abstractmethod
import pickle


class IAgent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def NAME(self): raise NotImplementedError

    @abstractmethod
    def __init__(self, input_size, output_size, training_mode, is_conv, load_filename, seed): raise NotImplementedError

    @abstractmethod
    def save_model(self, filename): raise NotImplementedError

    @abstractmethod
    def copy_from(self, model): raise NotImplementedError

    @abstractmethod
    def get_action(self, state): raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, done): raise NotImplementedError
