import sys
import pickle
from CuriousSamplePlanner.agent.planning_agent import PlanningAgent

# Environments
from CuriousSamplePlanner.tasks.two_block_stack import TwoBlocks
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
from CuriousSamplePlanner.tasks.four_block_stack import FourBlocks
from CuriousSamplePlanner.tasks.simple_2d import Simple2d

def visplan(load_id):
	# Find the plan and execute it
	path_filehandler = open("solution_data" + '/' + load_id + "/found_path.pkl", 'rb')
	graph_filehandler = open("solution_data" + '/' + load_id + "/found_graph.pkl", 'rb')
	expdict_filehandler = open("solution_data" + '/' + load_id + "/experiment_dict.pkl", 'rb')
	path = pickle.load(path_filehandler)
	graph = pickle.load(graph_filehandler)
	experiment_dict = pickle.load(expdict_filehandler)
	experiment_dict['render'] = True
	# Create the environment
	EC = getattr(sys.modules[__name__], experiment_dict["task"])
	env = EC(experiment_dict)

	agent = PlanningAgent(env)
	agent.execute_multistep_plan(path)

	print("Visualization Complete")



if __name__ == '__main__':
	load_id = str(sys.argv[1])
	visplan(load_id)

