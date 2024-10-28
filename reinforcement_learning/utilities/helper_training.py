import torch
# import math
import numpy as np
import copy
# from multiprocessing import Pool

# Tensordict modules
from tensordict.tensordict import TensorDictBase, TensorDict

# Data collection
from torchrl.collectors import SyncDataCollector

# Env
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import (
    set_exploration_type,
    _terminated_or_truncated,
    step_mdp,
)
from torchrl.envs.libs.vmas import VmasEnv

from vmas.simulator.utils import (
    X,
    Y,
)

# Utils
from matplotlib import pyplot as plt
from typing import Callable, Optional, Tuple, Callable, Optional, Union
from ctypes import byref

from matplotlib import pyplot as plt
import json
import os
import re


def get_model_name(parameters):
    # model_name = f"nags{parameters.n_agents}_it{parameters.n_iters}_fpb{parameters.frames_per_batch}_tfrms{parameters.total_frames}_neps{parameters.num_epochs}_mnbsz{parameters.minibatch_size}_lr{parameters.lr}_mgn{parameters.max_grad_norm}_clp{parameters.clip_epsilon}_gm{parameters.gamma}_lmbda{parameters.lmbda}_etp{parameters.entropy_eps}_mstp{parameters.max_steps}_nenvs{parameters.num_vmas_envs}"
    model_name = f"reward{parameters.episode_reward_mean_current:.2f}"

    return model_name

# def mpc_solve_parallel(j, agent_info, agent_actions_all, mpc, n_agents, slack_vars):

#     agent_actions = agent_actions_all[j,:]
#     agent_state = torch.stack([
#         agent_info["pos"][0,j,0],
#         agent_info["pos"][0,j,1],
#         agent_info["rot"][0,j,0],
#         agent_info["vel"][0,j,0],
#         agent_info["vel"][0,j,1]
#     ])

#     agent_ref_path = agent_info["ref"][0,j,:]
#     agent_ref_path_resized = agent_ref_path.view(-1, 2)
#     dx = agent_ref_path_resized[1:,0] - agent_ref_path_resized[:-1,0]
#     dy = agent_ref_path_resized[1:,1] - agent_ref_path_resized[:-1,1]
#     dx_first = torch.zeros(1)
#     dy_first = torch.zeros(1)
#     dx = torch.cat((dx_first, dx), 0)
#     dy = torch.cat((dy_first, dy), 0)
#     heading = torch.atan2(dy, dx)
#     velocity = torch.sqrt(dx**2 + dy**2)
#     vx = velocity * torch.cos(heading)
#     vy = velocity * torch.sin(heading)
#     agent_ref_path = torch.stack([agent_ref_path_resized[:,0], agent_ref_path_resized[:,1], heading, vx, vy], dim=-1)

#     other_agents_states = [[
#         agent_info["pos"][0,k,0],
#         agent_info["pos"][0,k,1],
#         agent_info["rot"][0,k,:],
#         agent_info["vel"][0,k,0],
#         agent_info["vel"][0,k,1]
#     ] for k in range(n_agents) if k != j]
#     other_agents_states = torch.tensor(other_agents_states)

#     mpc_agent_actions = mpc.solve(agent_state, agent_actions, agent_ref_path, n_agents-1, other_agents_states, slack_vars)
#     #print("[DEBUG] Agent ", j, " MPC actions: ", mpc_agent_actions)

#     return mpc_agent_actions

##################################################
## Custom Classes
##################################################  
class TransformedEnvCustom(TransformedEnv):
    """
    Slightly modify the function `rollout`, `_rollout_stop_early`, and `_rollout_nonstop` to enable returning a frame list to save evaluation video
    """
    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        callback: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = False,
        return_contiguous: bool = True,
        tensordict: Optional[TensorDictBase] = None,
        out=None,
        is_save_simulation_video: bool = False,
        mpc=True
    ):
        """Executes a rollout in the environment.

        The function will stop as soon as one of the contained environments
        returns done=True.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using :obj:`env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if ``True``, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is ``True``.
            auto_cast_to_device (bool, optional): if ``True``, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is ``False``.
            break_when_any_done (bool): breaks if any of the done state is True. If False, a reset() is
                called on the sub-envs that are done. Default is True.
            return_contiguous (bool): if False, a LazyStackedTensorDict will be returned. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.

        Returns:
            TensorDict object containing the resulting trajectory.

        The data returned will be marked with a "time" dimension name for the last
        dimension of the tensordict (at the ``env.ndim`` index).
        """
        # print("[DEBUG] new env.rollout")
        try:
            policy_device = next(policy.parameters()).device
        except (StopIteration, AttributeError):
            policy_device = self.device

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")
        if policy is None:

            policy = self.rand_action

        kwargs = {
            "tensordict": tensordict,
            "auto_cast_to_device": auto_cast_to_device,
            "max_steps": max_steps,
            "policy": policy,
            "policy_device": policy_device,
            "env_device": env_device,
            "callback": callback,
            "is_save_simulation_video": is_save_simulation_video,
            "mpc": mpc
        }
        if break_when_any_done:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_stop_early(**kwargs)
            else:
                tensordicts = self._rollout_stop_early(**kwargs)
        else:
            if is_save_simulation_video:
                tensordicts, frame_list = self._rollout_nonstop(**kwargs)
            else:
                tensordicts = self._rollout_nonstop(**kwargs)
                
        batch_size = self.batch_size if tensordict is None else tensordict.batch_size
        out_td = torch.stack(tensordicts, len(batch_size), out=out)
        if return_contiguous:
            out_td = out_td.contiguous()
        out_td.refine_names(..., "time")
        
        if is_save_simulation_video:
            return out_td, frame_list
        else:
            return out_td
        
    def _rollout_stop_early(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
        mpc
    ):
        tensordicts = []
        
        if is_save_simulation_video:
            frame_list = []
            
        for i in range(max_steps):
            if auto_cast_to_device:
                tensordict = tensordict.to(policy_device, non_blocking=True)
            tensordict = policy(tensordict)
            if auto_cast_to_device:
                tensordict = tensordict.to(env_device, non_blocking=True)
            tensordict = self.step(tensordict)
            tensordicts.append(tensordict.clone(False))

            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break
            tensordict = step_mdp(
                tensordict,
                keep_other=True,
                exclude_action=False,
                exclude_reward=True,
                reward_keys=self.reward_keys,
                action_keys=self.action_keys,
                done_keys=self.done_keys,
            )
            # done and truncated are in done_keys
            # We read if any key is done.
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=self.output_spec["full_done_spec"],
                key=None,
            )
            if any_done:
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)
                
        if is_save_simulation_video:
            return tensordicts, frame_list
        else:
            return tensordicts

    def _rollout_nonstop(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
        is_save_simulation_video,
        mpc
    ):
        tensordicts = []
        tensordict_ = tensordict

        if mpc:
            from mpc import MPC
            print("Using MPC")
            mpc_obj = MPC(
                Hp=60,
                Q = np.diag([1000, 1000, 1000, 1000, 1000]), 
                Qc = np.linspace(2900, 800, 60), #try 4000 or above next time
                Kj = 4,
                R = np.diag([1,1]),
                Ru = np.diag([750,50000]), 
                P = np.diag([500,500]), 
                nx = 5,
                nu = 2,
                lr = 0.08,
                wheelbase = 0.16,
                max_steering_angle = np.radians(35),
                max_speed = 1.0,
                car_w = 0.08,
                car_l = 0.16,
                dt = 0.05
            )
            # p = Pool()
            slack_vars = [[0.84] * (mpc_obj.Hp+1) for _ in range(self.n_agents-1)]
            prev_cmd = torch.zeros((self.batch_size[0], self.n_agents, 2))
            agent_ref_path = None
            number_of_agent_actions = 0
            total_diff_in_agent_actions = 0

        if is_save_simulation_video:
            frame_list = []

        
            
        for i in range(max_steps):
            print("[DEBUG] Timestep:", i)
            if not mpc: 
                if auto_cast_to_device:
                    tensordict_ = tensordict_.to(policy_device, non_blocking=True)
                tensordict_ = policy(tensordict_)
                if auto_cast_to_device:
                    tensordict_ = tensordict_.to(env_device, non_blocking=True)
            #############################
            ## MPC ##
            #############################

            if mpc:

                """
                In this section of the code, we will be using the MPC to generate the actions for the 
                agents
                1. Fetch MARL actions for several timesteps into the future (Hp)
                    1a. For each timestep, predict the future states of the agents using the kinematic 
                    bicycle model
                    1b. For each agent, calculate the rewards and observations
                    1c. Update a copy of the tensordict with the rewards and observations
                    1d. Pass the updated tensordict to the policy to get the actions for the next 
                    timestep
                    1e. Repeat the process until Hp is reached
                2. For each agent, extract the relevant state information and pass it,
                along with the sequence of MARL actions to the MPC. If applicable, also pass the
                predicted future states of the other agents to the MPC
                3. Solve the MPC problem for each agent
                4. Update the tensordict with the MPC actions
                5. Pass the updated tensordict to the environment
                6. Repeat the process for the next timestep until max_steps is reached
                """
                #generate sequence of MARL actions by using scenario to simulate actions
                env_copy = copy.deepcopy(self.scenario)
                tensordict_copy = tensordict_.clone().detach() 

                
                for k in range(mpc_obj.Hp): 

                    if auto_cast_to_device:
                        tensordict_copy = tensordict_copy.to(policy_device, non_blocking=True)
                    tensordict_copy = policy(tensordict_copy)
                    if auto_cast_to_device:
                        tensordict_copy = tensordict_copy.to(env_device, non_blocking=True)
                
                    if k == 0:
                        actions = [tensordict_copy["agents"]["action"]]
                    
                    if k > 0:
                        actions.append(tensordict_copy["agents"]["action"])

                    #process the actions, collect rewards and observations

                    for n in range(self.n_agents):
                        # Update the states of agents based on the actions according to the kinematic bicycle model
                        agent_x, agent_y = env_copy.world.agents[n].state.pos.split(1, dim=1)
                        agent_x_vel, agent_y_vel = env_copy.world.agents[n].state.vel.split(1, dim=1)

                        agent_state = [
                            agent_x,
                            agent_y,
                            env_copy.world.agents[n].state.rot,
                            torch.sqrt(agent_x_vel**2 + agent_y_vel**2),
                            torch.atan2(agent_y_vel, agent_x_vel)
                        ]

                        agent_actions = actions[k][:,n].split(1, dim=1)
                        
                        new_state = mpc_obj.predict_future_state(agent_state, agent_actions)
                        new_rot = torch.stack([new_state[2]]).view(self.batch_size[0],1)

                        env_copy.world.agents[n].state.pos = torch.stack([new_state[0], new_state[1]]).view(self.batch_size[0],2)
                        env_copy.world.agents[n].state.rot = new_rot
                        env_copy.world.agents[n].state.vel = torch.stack([new_state[3]*torch.cos(new_rot), new_state[3]*torch.sin(new_rot)]).view(self.batch_size[0],2)
                    n = 0

                    for n in range(self.n_agents):
                        # collect new info, rewards and observations for each agent
                        new_info = env_copy.info(env_copy.world.agents[n])
                        observations = env_copy.observation(env_copy.world.agents[n])
                        rewards = env_copy.reward(env_copy.world.agents[n])
                        
                        #plug the rewards and observations and new info into the tensordict
                        tensordict_copy["agents"]["episode_reward"][:,n,:] = rewards.clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["observation"][:,n,:] = observations.clone().detach().view(self.batch_size[0],-1)
                        tensordict_copy["agents"]["info"]["act_steer"][:,n,:] = new_info["act_steer"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["act_vel"][:,n,:] = new_info["act_vel"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["distance_left_b"][:,n,:] = new_info["distance_left_b"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["distance_ref"][:,n,:] = new_info["distance_ref"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["distance_right_b"][:,n,:] = new_info["distance_right_b"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["is_collision_with_agents"][:,n,:] = new_info["is_collision_with_agents"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["is_collision_with_lanelets"][:,n,:] = new_info["is_collision_with_lanelets"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["pos"][:,n,:] = new_info["pos"].clone().detach().view(self.batch_size[0],2)
                        # tensordict_copy["agents"]["info"]["ref"][:,n,:] = new_info["ref"].clone().detach().view(self.batch_size[0],40) 
                        tensordict_copy["agents"]["info"]["rot"][:,n,:] = new_info["rot"].clone().detach().view(self.batch_size[0],1)
                        tensordict_copy["agents"]["info"]["vel"][:,n,:] = new_info["vel"].clone().detach().view(self.batch_size[0],2)


                    tensordict_copy["done"] = torch.full((self.batch_size[0], 1), False)
                    tensordict_copy["terminated"] = torch.full((self.batch_size[0], 1), False)
                    
                    if k == mpc_obj.Hp-1: 
                        break

                
                del tensordict_copy, env_copy
                
                if auto_cast_to_device:
                    tensordict_ = tensordict_.to(policy_device, non_blocking=True)
                tensordict_ = policy(tensordict_)
                if auto_cast_to_device:
                    tensordict_ = tensordict_.to(env_device, non_blocking=True)

                agent_info = tensordict_["agents"]["info"].detach()

                agent_actions_all = tensordict_["agents"]["action"].detach()
                # print("[DEBUG] agent actions MARL: ", agent_actions_all)
                

                # #print("calling multiprocess")
                
                # results = p.starmap(mpc_solve_parallel, [(j, agent_info, agent_actions_all, mpc, self.n_agents, slack_vars) for j in range(self.n_agents)])
                # print(results)
                # #p.close()

                # tensordict_["agents"]["action"][0,:] = torch.stack(results)
                


                for m in range(self.batch_size[0]):
                    print(" [DEBUG] env:", m)
                    
                    predicted_future_states = None
                    for j in range(self.n_agents):
                        # agent_actions = agent_actions_all[j,:]
                        #print("[DEBUG] Agent ", j, " in iteration ", i, " actions are: ", agent_actions)

                        # get the state of the agent
                        agent_state = torch.stack([
                            agent_info["pos"][m,j,0],
                            agent_info["pos"][m,j,1],
                            agent_info["rot"][m,j,0],
                            torch.sqrt(torch.pow(agent_info["vel"][m,j,0],2) + torch.pow(agent_info["vel"][m,j,1],2)),
                            torch.atan2(agent_info["vel"][m,j,1], agent_info["vel"][m,j,0]) 
                        ])
                        #print("[DEBUG] Agent ", j, " in iteration ", i, " state is: ", agent_state)

                        #get the desired states (used when testing standalone MPC)
                        # agent_ref_path = agent_info["ref"][m,j,:].view(-1,2)

                        # # take every 5th point from the points in agent_ref_path
                        # agent_ref_path = agent_ref_path[::5]

                        # # infer the rest of the state information
                        # dx = torch.sub(agent_ref_path[1:,0], agent_ref_path[:-1,0])
                        # dy = torch.sub(agent_ref_path[1:,1], agent_ref_path[:-1,1])
                        # dx = torch.cat((torch.tensor([agent_ref_path[0,0] - agent_state[0]]), dx), 0)
                        # dy = torch.cat((torch.tensor([agent_ref_path[0,1] - agent_state[1]]), dy), 0)
                        # heading = torch.atan2(dy, dx)
                        # velocity =  torch.sqrt(torch.pow(dx,2) + torch.pow(dy,2)) * 7
                        # vx = velocity * torch.cos(heading)
                        # vy = velocity * torch.sin(heading)
                        # steering_angle = torch.atan2(vy, vx) 
                        # agent_ref_path = torch.stack([agent_ref_path[:,0], agent_ref_path[:,1], heading, velocity, steering_angle], dim=-1)
                        # print("[DEBUG] Agent ", j, " in iteration ", i, " ref path is: ", agent_ref_path)

                        # extract the sequence of MARL actions for the agent
                        agent_marl_u_seq = []
                        for action in actions:
                            agent_marl_u_seq.append(action[m,j,:].detach())
                        # agent_marl_u_seq = [agent_actions_all[m,j,:].detach()] (this line is used
                        # along with commenting the loop above when testing standalone MPC)

                        if (j > 2):
                            mpc_obj.Ru[0,0] += 65

                        
                        # solve the MPC problem for the agent
                        if (j == 0): 
                            mpc_agent_actions, predicted_future_states = mpc_obj.solve(agent_state, agent_marl_u_seq, agent_ref_path, slack_vars, None, prev_cmd[m,j,:])
                        else:
                            mpc_agent_actions, predicted_future_states = mpc_obj.solve(agent_state, agent_marl_u_seq, agent_ref_path, slack_vars, predicted_future_states, prev_cmd[m,j,:])

                        number_of_agent_actions += 1

                        # calculate the difference between marl action and mpc action (collected for analysis of results)
                        total_diff_in_agent_actions +=  torch.abs(agent_actions_all[m,j,:] - mpc_agent_actions) / agent_actions_all[m,j,:] 


                        # print(predicted_future_states)
                        # update the agent actions in the tensordict
                        tensordict_["agents"]["action"][m,j,:] = mpc_agent_actions
                        prev_cmd[m,j,:] = mpc_agent_actions

                print("[DEBUG] MPC actions for all agents are: ", tensordict_["agents"]["action"])
            ############################
            # End of MPC #
            ############################

            # print("[DEBUG] real timestep actions: ", tensordict_["agents"]["action"])
            tensordict, tensordict_ = self.step_and_maybe_reset(tensordict_)
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run

                if mpc:
                    print("MPC convergence ratio: ", (mpc_obj.total_converged_solutions / mpc_obj.total_solve_attempts)*100,"%")
                    print("MPC average iteration time: ", (mpc_obj.solve_attempt_time / mpc_obj.total_solve_attempts)*1000, "ms")
                    print("Average difference between MPC action and MARL action: ", (total_diff_in_agent_actions/number_of_agent_actions)*100, "%")
                break

            if callback is not None:
                if is_save_simulation_video:
                    frame = callback(self, tensordict)
                    frame_list.append(frame)
                else:
                    callback(self, tensordict)
                    
        # p.close()
        if is_save_simulation_video:
            return tensordicts, frame_list
        else:
            return tensordicts
        




class PriorityModule:
    def __init__(self, env: TransformedEnvCustom = None, mappo: bool = True):
        """
        Initializes the PriorityModule, which is responsible for computing the priority ordering of agents
        and their scores using a neural network policy. It also sets up a PPO loss module with an actor-critic
        architecture and GAE (Generalized Advantage Estimation) for reinforcement learning optimization.

        Parameters:
        -----------
        env : TransformedEnvCustom
            The environment containing the observation specifications and other scenario parameters.
        mappo : bool, optional
            Flag to indicate whether to use centralised learning in the critic (MAPPO). Default is True.
        """

        self.env = env
        self.parameters = self.env.scenario.parameters

        # Tuple containing the prefix keys relevant to the priority variables
        self.prefix_key = ("agents", "info", "priority")

        observation_key = get_priority_observation_key()

        policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec[observation_key].shape[-1],
                n_agent_outputs=2 * 1,  # 2 * n_actions_per_agents
                n_agents=self.parameters.n_agents,
                centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
                share_params=True,
                device=self.parameters.device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[observation_key],
            out_keys=[self.prefix_key + ("loc",), self.prefix_key + ("scale",)],
        )

        policy = ProbabilisticActor(
            module=policy_module,
            spec=UnboundedContinuousTensorSpec(),
            in_keys=[self.prefix_key + ("loc",), self.prefix_key + ("scale",)],
            out_keys=[self.prefix_key + ("scores",)],
            distribution_class=TanhNormal,
            distribution_kwargs={},
            return_log_prob=True,
            log_prob_key=self.prefix_key + ("sample_log_prob",),
        )  # we'll need the log-prob for the PPO loss

        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[observation_key].shape[
                -1
            ],  # Number of observations
            n_agent_outputs=1,  # 1 value per agent
            n_agents=self.parameters.n_agents,
            centralised=mappo,  # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
            share_params=True,  # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
            device=self.parameters.device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        critic = TensorDictModule(
            module=critic_net,
            in_keys=[
                observation_key
            ],  # Note that the critic in PPO only takes the same inputs (observations) as the actor
            out_keys=[self.prefix_key + ("state_value",)],
        )

        self.policy = policy
        self.critic = critic

        if self.parameters.prioritization_method.lower() == "marl":
            loss_module = ClipPPOLoss(
                actor=policy,
                critic=critic,
                clip_epsilon=self.parameters.clip_epsilon,
                entropy_coef=self.parameters.entropy_eps,
                normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
            )

            # Comment out advantage and value_target keys to use the same advantage for both base and priority loss modules
            loss_module.set_keys(  # We have to tell the loss where to find the keys
                reward=env.reward_key,
                action=self.prefix_key + ("scores",),
                sample_log_prob=self.prefix_key + ("sample_log_prob",),
                value=self.prefix_key + ("state_value",),
                # These last 2 keys will be expanded to match the reward shape
                done=("agents", "done"),
                terminated=("agents", "terminated"),
                # advantage=self.prefix_key + ("advantage",),
                # value_target=self.prefix_key + ("value_target",),
            )

            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.parameters.gamma,
                lmbda=self.parameters.lmbda,
            )  # We build GAE
            GAE = loss_module.value_estimator  # Generalized Advantage Estimation

            optim = torch.optim.Adam(loss_module.parameters(), self.parameters.lr)

            self.GAE = GAE
            self.loss_module = loss_module
            self.optim = optim

    def rank_agents(self, scores):
        """
        Ranks agents based on their priority scores.

        The method returns the indices of agents in descending order based on their scores.

        Parameters:
        -----------
        scores : Tensor
            A tensor containing the priority scores for each agent.

        Returns:
        --------
        ordered_indices : Tensor
            A tensor containing the indices of agents ordered by their priority scores in descending order.
        """
        # Remove the last dimension of size 1
        scores = scores.squeeze(-1)

        # Get the indices that would sort the tensor along the last dimension in descending order
        ordered_indices = torch.argsort(scores, dim=-1, descending=True)

        return ordered_indices

    def __call__(self, tensordict):
        """
        Computes the priority ordering of agents based on their scores and updates the tensordict.

        The method calls the priority actor to generate scores for each agent, ranks the agents
        based on those scores, and then updates the tensordict with the priority ordering.

        Parameters:
        -----------
        tensordict : TensorDict
            A dictionary-like object containing the data for the agents.

        Returns:
        --------
        tensordict : TensorDict
            The updated tensordict with the priority ordering of agents added.
        """

        # Call the priority actor and extract the scores key from the resulting tensordict
        scores = self.policy(tensordict)[self.prefix_key + ("scores",)]

        # Generate the priority ordering of agents
        priority_ordering = self.rank_agents(scores)

        tensordict[self.prefix_key + ("ordering",)] = priority_ordering

        # Return the tensordict with the priority ordering included
        return tensordict

    def compute_losses_and_optimize(self, data):
        """
        Computes the PPO loss (actor and critic losses) and performs backpropagation with gradient clipping.

        This method computes the combined loss (objective, critic, and entropy losses) from the loss module,
        checks for invalid gradients (NaN or infinite values), performs backpropagation, applies gradient clipping,
        and then steps the optimizer to update the model parameters.

        Parameters:
        -----------
        data : TensorDict
            A dictionary-like object containing the data for computing the losses.

        Returns:
        --------
        None
        """

        loss_vals = self.loss_module(data)

        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        assert not loss_value.isnan().any()
        assert not loss_value.isinf().any()

        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.parameters.max_grad_norm
        )  # Optional

        self.optim.step()
        self.optim.zero_grad()
class Parameters():
    def __init__(self,
                # General parameters
                n_agents: int = 4,          # Number of agents
                dt: float = 0.05,           # [s] sample time
                device: str = "cpu",        # Tensor device
                scenario_name: str = "road_traffic",    # Scenario name
                
                # Training parameters
                n_iters: int = 250,             # Number of training iterations
                frames_per_batch: int = 2**12, # Number of team frames collected per training iteration 
                                            # num_envs = frames_per_batch / max_steps
                                            # total_frames = frames_per_batch * n_iters
                                            # sub_batch_size = frames_per_batch // minibatch_size
                num_epochs: int = 60,       # Optimization steps per batch of data collected
                minibatch_size: int = 2**9,     # Size of the mini-batches in each optimization step (2**9 - 2**12?)
                lr: float = 2e-4,               # Learning rate
                lr_min: float = 1e-5,           # Minimum learning rate (used for scheduling of learning rate)
                max_grad_norm: float = 1.0,     # Maximum norm for the gradients
                clip_epsilon: float = 0.2,      # Clip value for PPO loss
                gamma: float = 0.99,            # Discount factor from 0 to 1. A greater value corresponds to a better farsight
                lmbda: float = 0.9,             # lambda for generalised advantage estimation
                entropy_eps: float = 1e-4,      # Coefficient of the entropy term in the PPO loss
                max_steps: int = 2**7,          # Episode steps before done
                total_frames: int = None,       # Total frame for one training, equals `frames_per_batch * n_iters`
                num_vmas_envs: int = None,      # Number of vectorized environments
                training_strategy: str = "4",  # One of {'1', '2', '3', '4'}. 
                                            # 1 for vanilla
                                            # 2 for vanilla with prioritized replay buffer
                                            # 3 for vanilla with challenging initial state buffer
                                            # 4 for mixed training
                episode_reward_mean_current: float = 0.00,  # Achieved mean episode reward (total/n_agents)
                episode_reward_intermediate: float = -1e3, # A arbitrary, small initial value
                
                is_prb: bool = False,       # # Whether to enable prioritized replay buffer
                scenario_probabilities = [1.0, 0.0, 0.0], # Probabilities of training agents in intersection, merge-in, or merge-out scenario
                
                # Observation
                n_points_short_term: int = 3,            # Number of points that build a short-term reference path

                is_partial_observation: bool = True,     # Whether to enable partial observation
                n_nearing_agents_observed: int = 2,      # Number of nearing agents to be observed (consider limited sensor range)

                # Parameters for ablation studies
                is_ego_view: bool = True,                           # Ego view or bird view
                is_apply_mask: bool = True,                         # Whether to mask distant agents
                is_observe_distance_to_agents: bool = True,         # Whether to observe the distance to other agents
                is_observe_distance_to_boundaries: bool = True,     # Whether to observe points on lanelet boundaries or observe the distance to labelet boundaries
                is_observe_distance_to_center_line: bool = True,    # Whether to observe the distance to reference path
                is_observe_vertices: bool = True,                         # Whether to observe the vertices of other agents (or center point)
                
                is_add_noise: bool = True,                          # Whether to add noise to observations
                is_observe_ref_path_other_agents: bool = False,     # Whether to observe the reference paths of other agents
                is_use_mtv_distance: bool = True,           # Whether to use MTV-based (Minimum Translation Vector) distance or c2c-based (center-to-center) distance.
                
                # Visu
                is_visualize_short_term_path: bool = True,  # Whether to visualize short-term reference paths
                is_visualize_lane_boundary: bool = False,   # Whether to visualize lane boundary
                is_real_time_rendering: bool = False,       # Simulation will be paused at each time step for a certain duration to enable real-time rendering
                is_visualize_extra_info: bool = True,       # Whether to render extra information such time and time step
                render_title: str = "",                     # The title to be rendered

                # Save/Load
                is_save_intermediate_model: bool = True,    # Whether to save intermediate model (also called checkpoint) with the hightest episode reward
                is_load_model: bool = False,                # Whether to load saved model
                is_load_final_model: bool = False,          # Whether to load the final model (last iteration)
                mode_name: str = None,
                where_to_save: str = "outputs/",            # Define where to save files such as intermediate models
                is_continue_train: bool = False,            # Whether to continue training after loading an offline model
                is_save_eval_results: bool = True,          # Whether to save evaluation results such as figures and evaluation outputs
                is_load_out_td: bool = False,               # Whether to load evaluation outputs
                
                is_testing_mode: bool = False,              # In testing mode, collisions do not terminate the current simulation
                is_save_simulation_video: bool = False,     # Whether to save simulation videos
                ):
        
        self.n_agents = n_agents
        self.dt = dt
        
        self.device = device
        self.scenario_name = scenario_name
        
        # Sampling
        self.n_iters = n_iters
        self.frames_per_batch = frames_per_batch
        
        if (frames_per_batch is not None) and (n_iters is not None):
            self.total_frames = frames_per_batch * n_iters

        # Training
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.lr_min = lr_min
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.max_steps = max_steps
        self.training_strategy = training_strategy
        
        if (frames_per_batch is not None) and (max_steps is not None):
            self.num_vmas_envs = frames_per_batch // max_steps # Number of vectorized envs. frames_per_batch should be divisible by this number,

        self.is_save_intermediate_model = is_save_intermediate_model
        self.is_load_model = is_load_model        
        self.is_load_final_model = is_load_final_model        
        
        self.episode_reward_mean_current = episode_reward_mean_current
        self.episode_reward_intermediate = episode_reward_intermediate
        self.where_to_save = where_to_save
        self.is_continue_train = is_continue_train

        # Observation
        self.is_partial_observation = is_partial_observation
        self.n_points_short_term = n_points_short_term
        self.n_nearing_agents_observed = n_nearing_agents_observed
        self.is_observe_distance_to_agents = is_observe_distance_to_agents
        
        self.is_testing_mode = is_testing_mode
        self.is_save_simulation_video = is_save_simulation_video
        self.is_visualize_short_term_path = is_visualize_short_term_path
        self.is_visualize_lane_boundary = is_visualize_lane_boundary
        
        self.is_ego_view = is_ego_view
        self.is_apply_mask = is_apply_mask
        self.is_use_mtv_distance = is_use_mtv_distance
        self.is_observe_distance_to_boundaries = is_observe_distance_to_boundaries
        self.is_observe_distance_to_center_line = is_observe_distance_to_center_line
        self.is_observe_vertices = is_observe_vertices
        self.is_add_noise = is_add_noise 
        self.is_observe_ref_path_other_agents = is_observe_ref_path_other_agents 

        self.is_save_eval_results = is_save_eval_results
        self.is_load_out_td = is_load_out_td
            
        self.is_real_time_rendering = is_real_time_rendering
        self.is_visualize_extra_info = is_visualize_extra_info
        self.render_title = render_title

        self.is_prb = is_prb
        self.scenario_probabilities = scenario_probabilities
        
        if (mode_name is None) and (scenario_name is not None):
            self.mode_name = get_model_name(self)
            
            
    def to_dict(self):
        # Create a dictionary representation of the instance
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_data):
        # Create an instance of the class from a dictionary
        return cls(**dict_data)

class SaveData():
    def __init__(self, parameters: Parameters, episode_reward_mean_list: [] = None):
        self.parameters = parameters
        self.episode_reward_mean_list = episode_reward_mean_list
    def to_dict(self):
        return {
            'parameters': self.parameters.to_dict(),  # Convert Parameters instance to dict
            'episode_reward_mean_list': self.episode_reward_mean_list
        }
    @classmethod
    def from_dict(cls, dict_data):
        parameters = Parameters.from_dict(dict_data['parameters'])  # Convert dict back to Parameters instance
        return cls(parameters, dict_data['episode_reward_mean_list'])



##################################################
## Helper Functions
##################################################
def get_path_to_save_model(parameters: Parameters):
    parameters.mode_name = get_model_name(parameters=parameters)
    
    PATH_POLICY = parameters.where_to_save + parameters.mode_name + "_policy.pth"
    PATH_CRITIC = parameters.where_to_save + parameters.mode_name + "_critic.pth"
    PATH_FIG = parameters.where_to_save + parameters.mode_name + "_training_process.pdf"
    PATH_JSON = parameters.where_to_save + parameters.mode_name + "_data.json"
    
    return PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON

def delete_files_with_lower_mean_reward(parameters:Parameters):
    # Regular expression pattern to match and capture the float number
    pattern = r'reward(-?[0-9]*\.?[0-9]+)_'

    # Iterate over files in the directory
    for file_name in os.listdir(parameters.where_to_save):
        match = re.search(pattern, file_name)
        if match:
            # Get the achieved mean episode reward of the saved model
            episode_reward_mean = float(match.group(1))
            if episode_reward_mean < parameters.episode_reward_intermediate:
                # Delete the saved model if its performance is worse
                os.remove(os.path.join(parameters.where_to_save, file_name))

def find_the_highest_reward_among_all_models(path):
    """This function returns the highest reward of the models stored in folder `parameters.where_to_save`"""
    # Initialize variables to track the highest reward and corresponding model
    highest_reward = float('-inf')
    
    pattern = r'reward(-?[0-9]*\.?[0-9]+)_'
    # Iterate through the files in the directory
    for filename in os.listdir(path):
        match = re.search(pattern, filename)
        if match:
            # Extract the reward and convert it to float
            episode_reward_mean = float(match.group(1))
            
            # Check if this reward is higher than the current highest
            if episode_reward_mean > highest_reward:
                highest_reward = episode_reward_mean # Update
                 
    return highest_reward


def save(parameters: Parameters, save_data: SaveData, policy=None, critic=None):    
    # Get paths
    PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON = get_path_to_save_model(parameters=parameters)
    
    # Save parameters and mean episode reward list
    json_object = json.dumps(save_data.to_dict(), indent=4) # Serializing json
    with open(PATH_JSON, "w") as outfile: # Writing to sample.json
        outfile.write(json_object)
    # Example to how to open the saved json file
    # with open('save_data.json', 'r') as file:
    #     data = json.load(file)
    #     loaded_parameters = Parameters.from_dict(data)

    # Save figure
    plt.clf()  # Clear the current figure to avoid drawing on the same figure in the next iteration
    plt.plot(save_data.episode_reward_mean_list)
    plt.xlabel("Training iterations")
    plt.ylabel("Episode reward mean")
    plt.tight_layout() # Set the layout to be tight to minimize white space !!! deprecated
    plt.savefig(PATH_FIG, format="pdf", bbox_inches="tight")
    # plt.savefig(PATH_FIG, format="pdf")

    # Save models
    if (policy != None) & (critic != None): 
        # Save current models
        torch.save(policy.state_dict(), PATH_POLICY)
        torch.save(critic.state_dict(), PATH_CRITIC)
        # Delete files with lower mean episode reward
        delete_files_with_lower_mean_reward(parameters=parameters)

    print(f"Saved model: {parameters.episode_reward_mean_current:.2f}.")

def compute_td_error(tensordict_data: TensorDict, gamma = 0.9):
    """
    Computes TD error.
    
    Args:
        gamma: discount factor
    """
    current_state_values = tensordict_data["agents"]["state_value"]
    next_rewards = tensordict_data.get(("next", "agents", "reward"))
    next_state_values = tensordict_data.get(("next", "agents", "state_value"))
    td_error = next_rewards + gamma * next_state_values - current_state_values # See Eq. (2) of Section B EXPERIMENTAL DETAILS of paper https://doi.org/10.48550/arXiv.1511.05952
    td_error = td_error.abs() # Magnitude is more interesting than the actual TD error
    
    td_error_average_over_agents = td_error.mean(dim=-2) # Cooperative agents
    
    # Normalize TD error to [-1, 1] (priorities must be positive)
    td_min = td_error_average_over_agents.min()
    td_max = td_error_average_over_agents.max()
    td_error_range = td_max - td_min
    td_error_range = max(td_error_range, 1e-3) # For numerical stability
    
    td_error_average_over_agents = (td_error_average_over_agents - td_min) / td_error_range
    td_error_average_over_agents = torch.clamp(td_error_average_over_agents, 1e-3, 1) # For numerical stability
    
    return td_error_average_over_agents


def opponent_modeling(
    tensordict,
    policy,
    n_nearing_agents_observed,
    nearing_agents_indices,
    noise_percentage: float = 0,
):
    """
    This function implements opponent modeling, inspired by [1].
    Each ego agent uses its own policy to predict the tentative actions of its surrounding agents, aiming to mitigate the non-stationarity problem.
    The ego agent appends these tentative actions to its observation stored in the input tensordict.

    Reference
        [1] Raileanu, Roberta, et al. "Modeling others using oneself in multi-agent reinforcement learning." International conference on machine learning. PMLR, 2018.
    """
    policy(tensordict)  # Run the policy to get tentative actions
    # Infer parameters
    n_agents = tensordict["agents"]["action"].shape[1]
    n_actions = tensordict["agents"]["action"].shape[2]

    batch_dim = tensordict.batch_size[0]
    device = tensordict.device

    actions_tentative = tensordict["agents"]["action"]

    if noise_percentage != 0:
        # Model inaccuracy to opponent modeling

        # A certain percentage of the maximum value as the noise standard diviation
        noise_std_speed = AGENTS["max_speed"] * noise_percentage
        noise_std_steering = math.radians(AGENTS["max_steering"]) * noise_percentage

        noise_actions = torch.cat(
            [
                torch.randn([batch_dim, n_agents, 1], device=actions_tentative.device)
                * noise_std_speed,
                torch.randn([batch_dim, n_agents, 1], device=actions_tentative.device)
                * noise_std_steering,
            ],
            dim=-1,
        )

        actions_tentative[:] += noise_actions

    for ego_agent in range(n_agents):
        for j in range(n_nearing_agents_observed):
            sur_agent = nearing_agents_indices[:, ego_agent, j]

            batch_indices = torch.arange(batch_dim, device=device, dtype=torch.int32)
            action_tentative_sur_agent = actions_tentative[batch_indices, sur_agent]

            # Update observation with tentative actions
            idx_action_start = (
                -(n_nearing_agents_observed - j) * 2
            )  # Start index of the action of surrounding agents in the observation (actions are appended at the end of the observation)
            idx_action_end = (
                idx_action_start + n_actions
            )  # End index of the action of surrounding agents in the observation (actions are appended at the end of the observation)
            if idx_action_end == 0:
                idx_action_end = None  # Avoid slicing with zero

            # Insert the tentative actions of the surrounding agents into each ago agent's observation
            tensordict["agents"]["observation"][
                :, ego_agent, idx_action_start:idx_action_end
            ] = action_tentative_sur_agent



def get_observation_key(parameters):
    return (
        ("agents", "observation")
        if not parameters.is_using_prioritized_marl
        else ("agents", "info", "base_observation")
    )


def get_priority_observation_key():
    return ("agents", "info", "priority_observation")


def prioritized_ap_policy(
    tensordict,
    policy,
    priority_module,
    nearing_agents_indices,
    prioritization_method,
    is_add_noise_to_actions: bool = False,
):
    """
    Implements prioritized action propagation (AP) for multiple agents.
    The function first generates a priority ordering using the provided priority module.
    Then, agents are processed in this priority order, where each agent computes its action
    and propagates it to lower-priority agents as part of their observation.

    Since the policy call generates actions for all agents in all environments at once,
    the function uses a mask to isolate the actions of the agent whose turn it is to act.
    These actions are progressively combined to form the full action tensor.

    Parameters:
    -----------
    tensordict : TensorDict
        A dictionary-like object that stores the data for all agents and environments.
    policy : Callable
        The policy function used to compute the actions for the agents.
    priority_module : Callable
        A module that computes the priority ordering for agents by wrapping the process
        of generating priority scores and ranking agents according to these scores.
    nearing_agents_indices : Tensor
        A tensor indicating the neighboring agents for each agent in each environment.

    Returns:
    --------
    tensordict : TensorDict
        The updated tensordict with combined actions and observations after prioritized action propagation.
    """
    base_observation_key = ("agents", "info", "base_observation")

    device = tensordict.device

    # Clone original observation
    original_obs = tensordict[base_observation_key].clone()

    # Infer parameters
    n_envs, n_agents, obs_dim, action_dim = (
        original_obs.shape[0],
        original_obs.shape[1],
        original_obs.shape[2],
        AGENTS["n_actions"],
    )

    if prioritization_method.lower() == "marl":
        # Generate priority ordering using the priority module
        priority_module(tensordict)
        # Extract priority ordering (shape: (n_envs, n_agents)) from tensordict
        priority_ordering = tensordict[priority_module.prefix_key + ("ordering",)]
    elif prioritization_method.lower() == "random":
        # Generate a random priority ordering
        priority_ordering = torch.stack(
            [torch.randperm(n_agents) for _ in range(n_envs)]
        )

    # Temporary tensors to store intermediate observations and combined results
    temp_obs = torch.zeros(n_envs, n_agents, obs_dim)
    combined_action = torch.zeros(n_envs, n_agents, action_dim)
    combined_loc = torch.zeros(n_envs, n_agents, action_dim)
    combined_sample_log_prob = torch.zeros(n_envs, n_agents)
    combined_scale = torch.zeros(n_envs, n_agents, action_dim)
    combined_obs = torch.zeros(n_envs, n_agents, obs_dim)

    # Loop through each step in the priority ordering
    for turn in range(n_agents):
        # Reset the observation
        tensordict[("agents", "info")].set("base_observation", original_obs)

        # Get the list of agents for the current turn based on priority
        current_turn_agents = priority_ordering[:, turn]

        # Create environment indices (from 0 to n_envs - 1)
        envs = torch.arange(n_envs)

        # Create a mask indicating which agents are acting in each environment
        mask = torch.zeros(n_envs, n_agents, dtype=torch.bool)
        mask[envs, current_turn_agents] = True

        # Prepare input for the policy by modifying the observations
        for env in range(n_envs):
            agent_idx = current_turn_agents[env]
            obs = tensordict[base_observation_key][env, agent_idx].clone()

            # Get the indices of neighboring agents
            current_turn_agent_neighbors = nearing_agents_indices[env, agent_idx].to(
                torch.int64
            )

            # Collect actions of the current agent's neighbors
            actions_so_far = combined_action[env, current_turn_agent_neighbors].view(-1)

            # Propagate the collected actions into the current agent's observation
            if is_add_noise_to_actions:
                noise_percentage = 0.05
                std_noise_speed = AGENTS["max_speed"] * noise_percentage
                std_noise_steering = (
                    math.radians(AGENTS["max_steering"]) * noise_percentage
                )
                noise = torch.cat(
                    [
                        std_noise_speed * torch.randn(action_dim, device=device),
                        std_noise_steering * torch.randn(action_dim, device=device),
                    ],
                    dim=-1,
                )
                obs[-len(actions_so_far) :] = actions_so_far + noise
            else:
                obs[-len(actions_so_far) :] = actions_so_far

            # Store the updated observation for the current agent in temp_obs
            temp_obs[env, agent_idx] = obs

        # Update the base observation with temp_obs for policy execution
        tensordict[("agents", "info")].set("base_observation", temp_obs)

        # Call the policy to generate actions for the agents
        policy(tensordict)

        # Use the mask to place the data into the correct positions in the combined tensors
        combined_action[mask] = tensordict[("agents", "action")][mask]
        combined_loc[mask] = tensordict[("agents", "loc")][mask]
        combined_sample_log_prob[mask] = tensordict[("agents", "sample_log_prob")][mask]
        combined_scale[mask] = tensordict[("agents", "scale")][mask]
        combined_obs[mask] = tensordict[base_observation_key][mask]

    # Write the combined actions back to the tensordict
    tensordict[("agents", "action")] = combined_action
    tensordict[("agents", "loc")] = combined_loc
    tensordict[("agents", "sample_log_prob")] = combined_sample_log_prob
    tensordict[("agents", "scale")] = combined_scale
    tensordict[base_observation_key] = combined_obs

    return tensordict
