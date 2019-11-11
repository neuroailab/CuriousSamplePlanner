class State():
	def __init__(self, num_objects, num_links):
		self.num_objects = num_objects
		self.num_links = num_links
		self.state_array = [0 for _ in range(self.num_objects*6+self.num_links)]

	@property
	def config_size(self):
		return len(self.state_array)