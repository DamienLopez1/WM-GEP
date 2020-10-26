WM-GEP:

- Minigrid World must be Installed

To run the experiment, from the command line in this directory:

	python WM-IMGEP.py

To change the experimental parameters:

	In line 49 of WM-IMGEP.py:
	RolloutGenerator Params: 
	- mdir: model directory, device, 
	- time_limit: number of samples in goal space before exploration, integer value e.g. 100
	- number_goals: number of goals to set over lifetime of agent, integer value e.g 200
	- Forward_model: 'M' = World Model, 'D' = Linear layers(do not use),
	- hiddengoals: True = Goals set in World Model, False = goals as observations(basically IMGEPs)
	- curiosityreward = True/False - not relevant in this implementation,
	- static: True = static VAE and HiddenVAE, False = constantly evolving VAE and HiddenVAE


	generator = RolloutGenerator(args.logdir, device, time_limit , number_goals,Forward_model,hiddengoals,curiosityreward,static)

To watch the agent explore, in WM-IMGEP.py:

	generator.rollout(None,render = True) 

Data generation:

	In generation_script, line 26:
		- data.mgw for generating data without preinitialised MDRNN and VAE

	python data/generation_script.py --rollouts 700 --rootdir D:\steps1000\datasets\mgw --threads 8

	
	Must train MDRNN and VAE using datagenerated from this.

	In generation_script, line 26, after pretraining MDRNN and VAE:

		- data.MGWHiddencollect 

	python data/generation_script.py --rollouts 700 --rootdir D:\steps1000\datasets\mgw --threads 8
	
		- produces hidden state for each transition.

	Use data from this line to train HiddenVAE.


Training MD-RNN, VAE and HiddenVAE:

VAE:
	python trainvae.py --epochs 1000 --logdir D:\steps1000

	-Line 65 and 67 must be modified to be:
		-x\datasets\mgw  (x is logdir)

MD-RNN (VAE must be trained):
	python  trainmdrnn.py --logdir D:\steps1000
		-Line 78 and 81 must be modified to be:
		-x\datasets\mgw (x is logdir)

HIddenVAE(MD-RNN and VAE must be trained):
	python trainvae.py --epochs 1000 --logdir D:\steps1000
	
	-Line 60 and 62 must be modified to be:
	-x\datasets\mgw (x is logdir)
	
