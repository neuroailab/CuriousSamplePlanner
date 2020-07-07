
#!/usr/bin/env python
from __future__ import print_function
import pybullet as p
import time
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
	Pose, Point, Euler, set_default_camera, stable_z, \
	BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import multiply, invert, get_pose, set_pose, get_movable_joints, \
	set_joint_positions, add_fixed_constraint_2, enable_real_time, disable_real_time, joint_controller, \
	enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
	end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
	inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
	step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach, ApplyForce
from CuriousSamplePlanner.scripts.utils import *

class Link(ApplyForce):
	def __init__(self, link_point1):
		self.sphere = None
		self.link_point1 = link_point1

	def iterator(self):
		self.sphere = p.loadURDF("./models/yellow_sphere.urdf", self.link_point1)
		return []

class PlanningAgent():

	def __init__(self, environment):
		self.environment = environment
		self.links = []
		if(self.environment.robot != None):
			self.robot = self.environment.robot
		else:
			self.add_arm(self.environment.arm_size)

	def execute_multistep_plan(self, plan):
		SIM_SPEED = 0.001
		VIS_SPEED = 0.01
		current_config = plan[0].config
		self.environment.set_state(current_config)
		start_world = WorldSaver()
		commands = []
		for i in range(1, len(plan)):
			if(plan[i].command != None):
				print(plan[i].command )
				# In case of detailed GMP
				commands.append(plan[i].command)
				self.execute(plan[i].command)
			else:
				# In case of non-detailed GMP
				macroaction = self.environment.macroaction
				macroaction.add_arm(self.robot)
				macroaction.gmp = True
				macroaction.teleport = False
				feasible, command = macroaction.execute(config = self.environment.get_current_config(), embedding = torch.squeeze(torch.tensor(np.array(plan[i].action))), sim=True)

				# Restore the state
				if(command is not None):
					self.execute(command, dt=SIM_SPEED)
					self.environment.set_state(plan[i].preconfig[0])
					self.environment.run_until_stable(hook = self.hook, dt=SIM_SPEED)
				commands.append(command)


		self.links = []
		self.environment.set_state(plan[0].config)
		

		for c_i , command in enumerate(commands):
			self.execute(command, dt=VIS_SPEED)
			for obj in self.environment.objects:
				p.resetBaseVelocity(obj, [0, 0, 0], [0, 0, 0])
			self.environment.set_state(plan[c_i+1].preconfig[0])
			self.environment.run_until_stable(dt=VIS_SPEED)

		for i in range(1000):
			p.stepSimulation()
			self.hook()
			time.sleep(0.01)



	def hook(self):
		for (link_object, link, link_transform, _) in self.links:
			if(link_object.sphere != None):
				lpos, lquat = p.getBasePositionAndOrientation(link)
				le = p.getEulerFromQuaternion(lquat)
				lpose = Pose(Point(*lpos), Euler(*le))
				set_pose(link_object.sphere, multiply(lpose, link_transform))

	def execute(self, command, dt=0.001):
		command.refine(num_steps=100).execute(time_step=dt, hook=self.hook)

	def hide_arm(self):
		p.removeBody(self.robot)

	def add_arm(self, arm_size):
		arm_size=1.1
		self.robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True, globalScaling=arm_size) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF




