#! /f/anaconda3/python
import getpass

user = getpass.getuser()
passwd = getpass.getpass()

print(user, passwd)
