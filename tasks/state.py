class State():
	def __init__(self, num_objects, num_static_objects, num_links):
		self.DOF = 6 # rotational and translational dof
		self.num_objects = num_objects
		self.num_static_objects = num_static_objects
		self.num_links = num_links
		self.state_array = [0 for _ in range((self.num_objects+self.num_static_objects)*self.DOF+self.num_links)]

	def set_position(self, object_index, x, y, z):
		assert object_index<self.num_objects and object_index>=0
		self.state_array[object_index*6+0] = x
		self.state_array[object_index*6+1] = y
		self.state_array[object_index*6+2] = z

	def set_rotation(self, object_index, x, y, z):
		assert object_index<self.num_objects and object_index>=0
		self.state_array[object_index*6+3] = x
		self.state_array[object_index*6+4] = y
		self.state_array[object_index*6+5] = z

	@property
	def positions(self):
		pos = []
		for i in range(self.num_objects):
			pos+=[i*self.DOF, i*self.DOF+1, i*self.DOF+2]
		return pos	

	@property
	def links(self):
		return list(range((self.num_objects+self.num_static_objects)*self.DOF, len(self.state_array)))
			
	@property 
	def config(self):
		return self.state_array

	@property
	def config_size(self):
		return len(self.state_array)