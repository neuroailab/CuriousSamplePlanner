<style>
	.column {
		float: left;
		width: 50%;
		padding: 5px;
	}

	/* Clear floats after image containers */
	.row::after {
		content: "";
		clear: both;
		display: table;
	}

	.tcolumn {
		float: left;
		width: 33.33%;
		padding: 5px;
	}

	/* Clear floats after image containers */
	.trow::after {
		content: "";
		clear: both;
		display: table;
	}
</style>



# What is the Curious Sample planner?

Many complex behaviors such as cleaning a kitchen, organizing a drawer, or cooking a meal require plans that are a combination of low-level geometric manipulation and high-level action sequencing.

Currently, there are two approaches to problems of this sort: planning and reinforcement learning. Unfortunately, these methods have problems that make them unusable for complex physical tasks that we were looking to solve with a flexible, multi-purpose algorithm.


<img src="./figs/problems.png" alt="Problems" style="width:100%">
	
To avoid these problems, we combine the strengths of deep reinforcement learning and task and motion planning to create an algorithm that can flexibly and efficiently find solutions to long-range planning problems through curious exploration. 

First, we will take a look at some of the types of problems we are trying to solve.

<div class="row">
	<div class="column">
		<h3>Block-Stacking</h3>
		<img src="./figs/an_5stack.gif" alt="Five-Stack" style="width:100%">
	</div>
	<div class="column">
		<h3>Ramp-Building</h3>
		<img src="./figs/an_ramp3.gif" alt="Ramp" style="width:100%">
	</div>
</div>
<div class="row">
	<div class="column">
		<h3>Tool-Construction</h3>
		<img src="./figs/an_bookshelf.gif" alt="Bookshelf" style="width:100%">
	</div>
	<div class="column">
		<h3>Pulley-Seesaw</h3>
		<img src="./figs/an_pulley.gif" alt="Pulley" style="width:100%">
	</div>
</div>

# Long-Range Planning Tasks

We aimed to create tasks that require temporally extended planning as well as low-level geometric manipulation. We constructed a suite of tasks with that objective in mind, and implemented those tasks using a modification of the [Pybullet-planning Library](https://github.com/caelan/ss-pybullet)

### Block-Stack

In this task, the robot is provided with a set of cubic blocks, and the robot's goal is to stack the blocks in a stable tower (in no particular order), but the robot is not provided with any reward at intermediate unsolved conditions. Block stacking is a commonly used task for evaluating planning algorithms because it requires detailed motor control and has a goal that is sparse in the state space.


<img src="./figs/t1.png" alt="Problem3" style="width:100%">


### Push-Away

In this task, the robot is provided with several large cubic blocks, a flat elongated block, and one additional smaller target object that can be of variable shape (e.g. a cube or a sphere). The objective is to somehow push the target object outside the directly reachable radius of the robotic arm. Depending on the situation, the solution to the task can be very simple or quite complex. For example, if the target object is cubic with high friction, it may be necessary for the robot to discover how to make construct a very steep ramp, such that dropping the object on it can roll it out of reach.

<img src="./figs/t2.png" alt="Problem3" style="width:100%">


### Bookshelf

In this task, the environment contains a bookshelf with a single book on it. 
The robot is also provided with two elongated rectangular-prism rods initially placed at random (reachable) locations in the environment. The goal is to knock the book off the bookshelf. However, the book and bookshelf are not only outside the reachable radius of the arm, but they are further than the combined length of the arm and a single rod. However, the robot can solve the task by combining the two rods in an end-to-end configuration using the link macro-action, and then using the combined object to dislodge the book.

<img src="./figs/t3.png" alt="Problem3" style="width:100%">


### Launch-Block

In this task, the environment contains a rope-and-pulley with one end of the rope connected to an anchor block on the floor and the other attached to a bucket that is suspended in mid-air. A seesaw balances evenly below the bucket, with a target block on the far end of the seesaw. The goal is to launch this target block into the air, above the seesaw surface. The robot could solve this task by gathering all blocks into the bucket and untying the anchor knot so that the bucket will descend onto the near end of the seesaw. However, due to the masses of the blocks and the friction in the seesaw, this can only happen when all five blocks are used.


<img src="./figs/t4.png" alt="Problem3" style="width:100%">


# What can CSP do?


In the process of planning, CSP discovers interesting solutions to complex problems, discovering the construction and use of tools and simple machines.  

A video illustrating some representative solutions can be found [here](https://youtu.be/7DSW8Dy9ADQ)

Sometimes CSP finds solutions that were unexpected and qualitatively different from those imagined by the authors.
As an example, we initially developed the Push-Away task with a spherical ball as a target object, hoping CSP would build a ramp and roll the ball out of the reachability zone. 
However, CSP instead found a simple solution consisting of aiming and dropping the ball precisely on the edge of another object, in order to get enough horizontal velocity to roll out of the reachability zone. 

In an attempt to avoid this rather trivial solution, we then switched out the ball for a block with a high coefficient of friction. Now, instead of always building a ramp, CSP sometimes made use of its link macro-action to fix the block to one end of the plank and orient it so that the block was held outside the reachability zone. Once the link macro-action was disabled, ramp-building behavior robustly emerged.

These examples show how CSP is not biased in the direction of any one solution by reward curricularization, but rather can discover many qualitatively distinct solutions to a single objective.

<div class="trow">
	<div class="tcolumn">
		<img src="./figs/ramp1.gif" alt="Ramp1" style="width:100%">
	</div>
	<div class="tcolumn">
		<img src="./figs/ramp2.gif" alt="Ramp2" style="width:100%">
	</div>
	<div class="tcolumn">
		<img src="./figs/ramp3.gif" alt="Ramp3" style="width:100%">
	</div>
</div>




# How does CSP Work?


<img src="./figs/arch.gif" alt="CSP Architecture" style="width:100%">


At its core, CSP is an algorithm for efficiently building a search tree over the state space using parameterized macro-actions.
CSP is comprised of four main modules. The action selection networks include an actor-network and a critic-network, which learn to select macro-actions and choose parameters of that macro-action given a particular state. 
The action selection networks have two primary functions: maximizing curiosity in action selection and avoiding infeasible macro-actions. The networks are trained using actor-critic reinforcement learning. The networks select feasible actions that maximize the novelty signal, leading to actions that result in novel configurations or dynamics. The actor-network outputs a continuous (real-valued) vector which is translated into a macro-action with both discrete and continuous parameters. The forward dynamics module takes a state and an action primitive simulates forward a fixed time and returns the resulting state. This forward dynamics module is used by a geometric planning module to convert macro-actions into feasible sequences of motor primitives. 
Finally, the curiosity module is a neural network that takes states as inputs and returns a curiosity score, with learnable parameters.

# Why do we need CSP?

We compared the performance of CSP to several reinforcement learning and planning baselines using a single metric: the number of steps taken in the environment.

* It is also important to note that it is impossible to compare our algorithm to TAMP because of the manual prespecification of action effects necessary for TAMP to function.



<img src="./figs/quantitative.png" alt="Quant" style="width:100%">


Overall, we found that CSP was dramatically more effective than the baselines at solving all four task types.
The control algorithms (including the CSP-No Curiosity baseline) were sometimes able to discover solution in the simplest cases (e.g. the 3-Block task).

However, they were substantially more variable in terms of the number of samples needed; and they largely failed to achieve solutions in more complex cases within a maximum sample limit.

The failure of the random and vanilla PPO/A2C baseliness is not surprising: the tasks we chose here are representative of the kind of long-range planning problems that typically are extremely difficult for standard DRL approaches.

# Why does CSP Work?

We can qualitatively compare the exploration patterns of different planning algorithms by examining the graphs generated during the planning process.  

The following figure gives an example of how multiple baseline algorithms explore the state space in comparison to CSP. While RRT (a geometric motion planning method) and random exploration are sometimes able to achieve two-block towers, they rarely progress to taller stacks because of the unlikeliness in the state space. On the other hand, because CSP biases the search toward unlikely states in the configuration space, tower configurations are oversampled and as a result, higher-stacked towers are explored.


<img src="./figs/qualitative.png" alt="Qual" style="width:100%">

In the process of planning, CSP discovers interesting solutions to complex problems, discovering the construction and use of tools and simple machines.

We can also visualize some of the high-novelty nodes in the search tree that were not solutions. These nodes were heavily explored during the search process due to their novelty but didn’t lead to the goal. 

While these were not the goal state, they were reasonable avenues to explore in the process of finding the goal.

<img src="./figs/bloopers.png" alt="Qual" style="width:100%">


# Learn more

You can find the code for this algorithm [here](https://github.com/neuroailab/CuriousSamplePlanner) and the paper can be viewed on [arXiv](https://arxiv.org/abs/2004.10876)

