"""
Copyright John Persano 2017

File name:      log.py
Description:    A log utility for colored Python printing.
Commit history:
                - 03/19/2017: Initial version
"""


class Log:
    """
    Log utility that will print messages according to a message level.
    """

    class Colors:
        """
        Color palette for the Log class.
        This should not be used directly.
        """
        RED = '\x1b[31m'
        YELLOW = '\x1b[33m'
        BLUE = '\x1b[34m'
        NORMAL = '\x1b[0m'

    @staticmethod
    def warning(text):
        """
        Prints yellow colored text.
        :param text: The text to print
        :return: None
        """
        print(Log.Colors.YELLOW + "WARNING: " + str(text) + Log.Colors.NORMAL)

    @staticmethod
    def error(text):
        """
        Prints red colored text.
        :param text: The text to print
        :return: None
        """
        print(Log.Colors.RED + "ERROR: " + str(text) + Log.Colors.NORMAL)

    @staticmethod
    def debug(text):
        """
        Prints red colored text.
        :param text: The text to print
        :return: None
        """
        print(Log.Colors.BLUE + "DEBUG: " + str(text) + Log.Colors.NORMAL)

    @staticmethod
    def info(text):
        """
        Prints white colored text.
        :param text: The text to print
        :return: None
        """
        print(str(text))
