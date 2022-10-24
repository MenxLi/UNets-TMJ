from io import TextIOWrapper
import sys

class Logger():
    # https://cloud.tencent.com/developer/article/1643418
    def __init__(self, file_obj: TextIOWrapper, write_to_terminal: bool = True):
        self.terminal = sys.stdout
        self.log = file_obj
        self.write_terminal = write_to_terminal
 
    def write(self, message):
        if self.write_terminal:
            self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        if self.write_terminal:
            self.terminal.flush()