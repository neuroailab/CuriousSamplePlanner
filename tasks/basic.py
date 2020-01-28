import random

class Basic:
	def __init__(self):
		
		self.radius = 0.05
		self.center_x = 0
		self.center_y = 0
		self.reset()
		

	def step(self, action):


	def reset(self):
		self.center_x = random.uniform(-1, 1)
		self.center_y = random.uniform(-1, 1)
		

		