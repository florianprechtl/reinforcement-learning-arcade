import sys

class Logger(object):
	def __init__(self):
		self.data = ""
		self.terminal = sys.stdout

	def write(self, message):
		self.data = "".join([self.data, message]) #self.data += message
		self.terminal.write(message)

	def flush(self):
		pass
