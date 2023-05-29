from abc import ABC, abstractmethod


class Ezra(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read_and_answer(self, content: str, question: str):
        pass
