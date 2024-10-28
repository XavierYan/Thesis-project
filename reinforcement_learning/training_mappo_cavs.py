# Adapted from https://pytorch.org/rl/tutorials/multiagent_ppo.html
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import time
import multiprocessing

from termcolor import colored, cprint
import sys
import os
# Torch
import torch
import matplotlib
matplotlib.rcParams['text.usetex'] = False



from Vectornet.vectornet_actor import VectorNet,MultiAgentVectorNetActor, CustomSequential,MultiAgentVectorNetCritic

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data import TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum
from torchrl.envs.utils import (
    check_env_specs,
)
# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from tqdm import tqdm

import os, sys

import matplotlib.pyplot as plt

# Scientific plotting
import scienceplots # Do not remove (https://github.com/garrettj403/SciencePlots)
plt.rcParams.update({'figure.dpi': '100'}) # Avoid DPI problem (https://github.com/garrettj403/SciencePlots/issues/60)
plt.style.use(['science','ieee']) # The science + ieee styles for IEEE papers (can also be one of 'ieee' and 'science' )
# print(plt.style.available) # List all available style

from torchrl.envs.libs.vmas import VmasEnv

# Import custom classes
from utilities.helper_training import Parameters, SaveData, TransformedEnvCustom, get_path_to_save_model, find_the_highest_reward_among_all_models, save, compute_td_error

from scenarios.round import ScenarioRoadTraffic

# Reproducibility
torch.manual_seed(0)


def mappo_cavs(parameters: Parameters):
    scenario = ScenarioRoadTraffic()
    device = torch.device("cuda")
    scenario.parameters = parameters

    # Using multi-threads to handle file writing
    # pool = ThreadPoolExecutor(128)

    env = VmasEnv(
        scenario=scenario,
        num_envs=parameters.num_vmas_envs,
        continuous_actions=True,  # VMAS supports both continuous and discrete actions
        max_steps=parameters.max_steps,
        device=parameters.device,
        # Scenario kwargs
        n_agents=parameters.n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    )
    
    save_data = SaveData(
        parameters=parameters,
        episode_reward_mean_list=[],
    )

    # print("env.full_action_spec:", env.full_action_spec, "\n")
    # print("env.full_reward_spec:", env.full_reward_spec, "\n")
    # print("env.full_done_spec:", env.full_done_spec, "\n")
    # print("env.observation_spec:", env.observation_spec, "\n")

    # print("env.action_keys:", env.action_keys, "\n")
    # print("env.reward_keys:", env.reward_keys, "\n")
    # print("env.done_keys:", env.done_keys, "\n")

    env = TransformedEnvCustom(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    check_env_specs(env)
    # policy_net = torch.nn.Sequential(
    #     VectorNet(),
    #     NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a `loc` and a non-negative `scale``, used as parameters for a normal distribution (mean and standard deviation)
    # )

    # # print("policy_net:", policy_net, "\n")

    # policy_module = TensorDictModule(
    #     policy_net,
    #     # in_keys=[("agents", "observation","ego_state" ), ("agents", "observation","ego_reference"),("agents", "observation", "neighbour_states"),
    #     #        ("agents", "observation","neighbour_reference"), ("agents", "observation","lane_boundary" )],
    #     in_keys=[("agents", "observation","ego_state" )],
    #     out_keys=[("agents", "loc"), ("agents", "scale")], # represents the parameters of the policy distribution for each agent
    # )
 
    vector_net = VectorNet()

    mavector_net = MultiAgentVectorNetActor(n_agents=parameters.n_agents, network = vector_net)
    normal_param_extractor = NormalParamExtractor()
    actor_net = CustomSequential(mavector_net, normal_param_extractor)
    actor_net = actor_net.to(device)


    policy_net = TensorDictModule(
        module=actor_net,
        in_keys=[
            ("agents", "observation", "ego_state"), 
            ("agents", "observation", "ego_reference"), 
            ("agents", "observation", "neighbour_states"),
            ("agents", "observation", "neighbour_reference"), 
            ("agents", "observation", "lane_boundary"),
        ],
        out_keys=[("agents", "loc"), ("agents", "scale")]
    )

    # Use a probabilistic actor allows for exploration
    policy = ProbabilisticActor(
        module=policy_net,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[env.action_key].space.low,
            "max": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"), # log probability favors numerical stability and gradient calculation
    )  # we'll need the log-prob for the PPO loss
    print(env.unbatched_action_spec[env.action_key].space.low)
    print(env.unbatched_action_spec[env.action_key].space.high)
    mappo = True  # IPPO (Independent PPO) if False

    critic_net = MultiAgentVectorNetCritic(n_agents=parameters.n_agents, network = vector_net)
    # critic_net = MultiAgentMLP(
    #     n_agent_inputs=env.observation_spec["agents", "observation","critic_obs"].shape[-1], # Number of observations
    #     n_agent_outputs=1,  # 1 value per agent
    #     n_agents=env.n_agents,
    #     centralised=mappo, # If `centralised` is True (which may help overcome the non-stationary problem in MARL), each agent will use the inputs of all agents to compute its output (n_agent_inputs * n_agents will be the number of inputs for one agent). Otherwise, each agent will only use its data as input.
    #     share_params=True, # If `share_params` is True, the same MLP will be used to make the forward pass for all agents (homogeneous policies). Otherwise, each agent will use a different MLP to process its input (heterogeneous policies).
    #     device=parameters.device,
    #     depth=2,
    #     num_cells=256,
    #     activation_class=torch.nn.Tanh,
    # )
    critic_net = critic_net.to(device)

    
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[
            ("agents", "observation", "ego_state"), 
            ("agents", "observation", "ego_reference"), 
            ("agents", "observation", "neighbour_states"),
            ("agents", "observation", "neighbour_reference"), 
            ("agents", "observation", "lane_boundary"),
        ],
        out_keys=[("agents", "state_value")],
    )

    # Check if the directory defined to store the model exists and create it if not
    if not os.path.exists(parameters.where_to_save):
        os.makedirs(parameters.where_to_save)
        print(colored("[INFO] Created a new directory to save the trained model:", "black"), colored(f"{parameters.where_to_save}", "blue"))

    # Load an existing model or train a new model?

    if parameters.is_load_imitation == True:
        checkpoint = torch.load('trained_model/IL/reward85.71_policy.pth')
        model_state_dict = checkpoint['model_state_dict']
        mavector_net.load_state_dict(model_state_dict)
        parameters.is_load_model == False


    if parameters.is_load_model:
        if not parameters.test_imitation:
            # Load the model with the highest reward in the folder `parameters.where_to_save`
            highest_reward = find_the_highest_reward_among_all_models(parameters.where_to_save)
            # highest_reward = 88.25
    

            parameters.episode_reward_mean_current = highest_reward # Update the parameter so that the right filename will be returned later on 
            if highest_reward is not float('-inf'):
                if parameters.is_load_final_model:
                    policy.load_state_dict(torch.load(parameters.where_to_save + "final_policy.pth"))
                    print(colored("[INFO] Loaded the final model (instead of the intermediate model with the highest episode reward)", "red"))
                else:
                    PATH_POLICY, PATH_CRITIC, PATH_FIG, PATH_JSON = get_path_to_save_model(parameters=parameters)
                    # Load the saved model state dictionaries
                    policy.load_state_dict(torch.load(PATH_POLICY))
                    print(colored("[INFO] Loaded the intermediate model with the highest episode reward", "blue"))
            else:
                raise ValueError("There is no model stored in '{parameters.where_to_save}', or the model names stored here are not following the right pattern.")

            if not parameters.is_continue_train:
                print(colored("[INFO] Training will not continue.", "blue"))
                return env, policy, parameters
            else:
                print(colored("[INFO] Training will continue with the loaded model.", "red"))
                critic.load_state_dict(torch.load(PATH_CRITIC))
        
        else:
            checkpoint = torch.load('/home/xavier/project/thesis/code/src/model_ckpt/0627/model_epoch45.pth')
            model_state_dict = checkpoint['model_state_dict']
            mavector_net.load_state_dict(model_state_dict)
            return env, policy, parameters
    


    collector = SyncDataCollector(
        env,
        policy,
        device=parameters.device,
        storing_device=parameters.device,
        frames_per_batch=parameters.frames_per_batch,
        total_frames=parameters.total_frames,
    )

    if parameters.is_prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=1.1,
            storage=LazyTensorStorage(
                parameters.frames_per_batch, device=parameters.device
            ),
            batch_size=parameters.minibatch_size,
            priority_key="td_error",
        )
    else:
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                parameters.frames_per_batch, device=parameters.device
            ),  # We store the frames_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=parameters.minibatch_size,  # We will sample minibatches of this size
        )

    loss_module = ClipPPOLoss(
        actor=policy,
        critic=critic,
        clip_epsilon=parameters.clip_epsilon,
        entropy_coef=parameters.entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=parameters.gamma, lmbda=parameters.lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator # Generalized Advantage Estimation 

    optim = torch.optim.Adam(loss_module.parameters(), parameters.lr)

    pbar = tqdm(total=parameters.n_iters, desc="epi_rew_mean = 0")

    episode_reward_mean_list = []
    v_loss_list = []
    loss_objective_list = []
    entropy_list = []
    iteration_counter = 0 

    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_params,
                target_params=loss_module.target_critic_params,
            )  # Compute GAE and add it to the data

        # Update sample priorities
        if parameters.is_prb:
            td_error = compute_td_error(tensordict_data, gamma=0.9)
            tensordict_data.set(("td_error"), td_error)  # Adding TD error to the tensordict_data
            
            assert tensordict_data["td_error"].min() >= 0, "TD error must be greater than 0"
            
        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)
        # replay_buffer.update_tensordict_priority() # Not necessary, as priorities were updated automatically when calling `replay_buffer.extend()`

        for _ in range(parameters.num_epochs):
            # print("[DEBUG] for _ in range(parameters.num_epochs):")
            for _ in range(parameters.frames_per_batch // parameters.minibatch_size):
                # sample a batch of data
                mini_batch_data, info = replay_buffer.sample(return_info=True)

                loss_vals = loss_module(mini_batch_data)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                v_loss_list.append(loss_vals["loss_critic"].item())
                loss_objective_list.append(loss_vals["loss_objective"].item())
                entropy_list.append(loss_vals["loss_entropy"].item())
                # Access and use log_prob

                # if ("agents", "sample_log_prob") in mini_batch_data:
                #     log_probs = mini_batch_data[("agents", "sample_log_prob")]
                #     # Calculate KL divergence or other metrics using log_probs
                #     # Example: KL divergence between old and new log_probs
                #     if "old_log_prob" in mini_batch_data:
                #         kl_divergence = torch.mean(mini_batch_data["old_log_prob"] - log_probs)
                #         save_data.kl_divergence_list.append(kl_divergence.item())
                



                assert not loss_value.isnan().any()
                assert not loss_value.isinf().any()

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), parameters.max_grad_norm
                )  # Optional

                optim.step()
                optim.zero_grad()
                
                if parameters.is_prb:
                    # Recalculate loss
                    with torch.no_grad():
                        GAE(
                            mini_batch_data,
                            params=loss_module.critic_params,
                            target_params=loss_module.target_critic_params,
                        )
                    # Recalculate the TD errors of the sampled minibatch with updated model weights and update priorities in the buffer
                    new_td_errors = compute_td_error(mini_batch_data, gamma=0.9)
                    mini_batch_data.set("td_error", new_td_errors)
                    replay_buffer.update_tensordict_priority(mini_batch_data)
        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        episode_reward_mean = round(episode_reward_mean, 2)
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean:.2f}", refresh=False)

        # env.scenario.iter = pbar.n # A way to pass the information from the training algorithm to the environment
        
        if parameters.is_save_intermediate_model:
            # Update the current mean episode reward
            parameters.episode_reward_mean_current = episode_reward_mean
            save_data.episode_reward_mean_list = episode_reward_mean_list
            save_data.v_loss_list = v_loss_list
            save_data.loss_objective_list = loss_objective_list
            save_data.entropy_list = entropy_list


            # if episode_reward_mean > parameters.episode_reward_intermediate:
            #     # Save the model if it improves the mean episode reward sufficiently enough
            #     parameters.episode_reward_intermediate = episode_reward_mean
            #     save(parameters=parameters, save_data=save_data, policy=policy, critic=critic)

            # else:
            #     # Save only the mean episode reward list and parameters
            #     parameters.episode_reward_mean_current = parameters.episode_reward_intermediate
            #     save(parameters=parameters, save_data=save_data, policy=None, critic=None)


            if episode_reward_mean > parameters.episode_reward_intermediate:
                parameters.episode_reward_intermediate = episode_reward_mean  
                save(parameters=parameters, save_data=save_data, policy=policy, critic=critic)  
                iteration_counter = 0 
            else:
                iteration_counter += 1 
                if iteration_counter % 10 == 0:  
                    save(parameters=parameters, save_data=save_data, policy=policy, critic=critic)
                    parameters.episode_reward_mean_current = parameters.episode_reward_intermediate
                    save(parameters=parameters, save_data=save_data, policy=None, critic=None)  
                else:
                    parameters.episode_reward_mean_current = parameters.episode_reward_intermediate
                    save(parameters=parameters, save_data=save_data, policy=None, critic=None)  
                    



        # Learning rate schedule
        for param_group in optim.param_groups:
            # Linear decay to lr_min
            lr_decay = (parameters.lr - parameters.lr_min) * (1 - (pbar.n / parameters.n_iters))
            param_group['lr'] = parameters.lr_min + lr_decay
            if (pbar.n % 10 == 0):
                print(f"Learning rate updated to {param_group['lr']}.")
                
        pbar.update()
        
    # Save the final model
    torch.save(policy.state_dict(), parameters.where_to_save + "final_policy.pth")
    torch.save(critic.state_dict(), parameters.where_to_save + "final_critic.pth")
    print(colored("[INFO] All files have been saved under:", "black"), colored(f"{parameters.where_to_save}", "red"))
    # plt.show()
    
    return env, policy, parameters

if __name__ == "__main__":
    scenario_name = "road_traffic" # road_traffic, path_tracking, obstacle_avoidance
    
    parameters = Parameters(
        n_agents=6,
        dt=0.04, # [s] sample time 
        device="cuda:0" if not torch.cuda.is_available() else "cuda:0",  # The divice where learning is run
        scenario_name=scenario_name,
        
        # Training parameters
        n_iters=250, # Number of sampling and training iterations (on-policy: rollouts are collected during sampling phase, which will be immediately used in the training phase of the same iteration),
        frames_per_batch=2**13, # Number of team frames collected per training iteration 
                                    # num_envs = frames_per_batch / max_steps
                                    # total_frames = frames_per_batch * n_iters
                                    # sub_batch_size = frames_per_batch // minibatch_size
        num_epochs=5,          # Optimization steps per batch of data collected,
        minibatch_size=2**9,    # Size of the mini-batches in each optimization step (2**9 - 2**12?),
        lr=2e-4,                # Learning rate,
        lr_min=1e-5,            # Min Learning rate,
        max_grad_norm=1.0,      # Maximum norm for the gradients,
        clip_epsilon=0.2,       # Clip value for PPO loss,
        gamma=0.99,             # Discount factor from 0 to 1. A greater value corresponds to a better farsight.
        lmbda=0.9,              # lambda for generalised advantage estimation,
        entropy_eps=1e-4,       # Coefficient of the entropy term in the PPO loss,
        max_steps=2**8,         # Episode steps before done
        training_strategy='2',  # One of {'1', '2', '3', '4'}. 
                                    # 1 for vanilla
                                    # 2 for vanilla with prioritized replay buffer
                                    # 3 for vanilla with challenging initial state buffer
                                    # 4 for our
        is_save_intermediate_model=True, # Is this is true, the model with the highest mean episode reward will be saved,
        
        episode_reward_mean_current=0.00,
        

        is_load_imitation = False,   #use imitation model to continue training
        is_load_model=False,        # Load offline model if available. The offline model in `where_to_save` whose name contains `episode_reward_mean_current` will be loaded
        is_load_final_model=False,  # Whether to load the final model instead of the intermediate model with the highest episode reward
        is_continue_train=True,    # If offline models are loaded, whether to continue to train the model
        mode_name=None, 
        episode_reward_intermediate=-1e3, # The initial value should be samll enough
        
        where_to_save=f"trained_model/IL+RL/", # folder where to save the trained models, fig, data, etc.

        # Scenario parameters
        is_partial_observation=True,
        n_points_short_term=15,
        n_nearing_agents_observed=4,
        
        is_testing_mode=False,
        is_visualize_short_term_path=True,
        
        is_save_eval_results=True,
        
        is_prb=False,       # Whether to enable prioritized replay buffer
        scenario_probabilities=[1.0, 0.0, 0.0],
        
        is_use_mtv_distance=False,

        # Ablation studies
        is_ego_view=True,                   # Eago view or bird view
        is_apply_mask=True,                 # Whether to mask distant agents
        is_observe_distance_to_agents=True,      
        is_observe_vertices=True,
        is_observe_distance_to_boundaries=True,  
        is_observe_distance_to_center_line=True,
        
        is_add_noise=True,
        is_observe_ref_path_other_agents=False,
    )
    
    if parameters.training_strategy == "2":
        parameters.is_prb=True
        
    env, policy, parameters = mappo_cavs(parameters=parameters)

    # Evaluate the model
    # with torch.no_grad():
    #     out_td = env.rollout(
    #         max_steps=parameters.max_steps,
    #         policy=policy,
    #         callback=lambda env, _: env.render(),
    #         auto_cast_to_device=True,
    #         break_when_any_done=False,
    #     )
