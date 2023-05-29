# This is a sample Python script.
from ezra.randeng.Randeng import Randeng


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def sample():
    """
    Run `git clone https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-QA-Chinese` first,
    store the model repo into `./models`.
    :return:
    """


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    USE IDEA-CCNL/Randeng-T5-784M-QA-Chinese
    """
    ezra = Randeng('./models/Randeng-T5-784M-QA-Chinese')
    answers = ezra.read_and_answer(
        """
        就如　神从创立世界以前，在基督里拣选了我们，使我们在他面前成为圣洁，无有瑕疵；
        又因爱我们，就按着自己意旨所喜悦的，预定我们藉着耶稣基督得儿子的名分，
        使他荣耀的恩典得着称赞。这恩典是他在爱子里所赐给我们的。  
        """,
        """
        神为什么拣选我们？
        """
    )
    print(answers)
