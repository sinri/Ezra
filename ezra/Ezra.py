from abc import ABC, abstractmethod
from typing import List


class Ezra(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read_and_answer(self, content: str, question: str) -> List[str]:
        pass
