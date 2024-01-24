import rospy
import scipy
import sys
from typing import List
from std_msgs.msg import Float32, Int16
from gym import wrappers, spaces
from gym.envs.box2d.car_racing import CarRacing
from time import time
import numpy as np
import torch as th
import matplotlib.pyplot as plt

import pandas as pd
from stable_baselines3 import DQNCopilot
from stable_baselines3.dqn_copilot.policies import CnnPolicyCopilot 
from queue import Queue
from scipy.signal import savgol_filter


STATE_H = 96
STATE_W = 96

#convert discrete action to continuous action
def disc2cont(action):
    if action == 0:
        action = [0,  0.3, 0.05]  # "NOTHING"
    elif 1 <= action <= 5:
        action = [round((-0.2*action),1),  0.3, 0.05]
    elif 6 <= action <= 10:
        action = [round(0.2*(action-5),1),  0.3, 0.05]
    return action


def cont2disc_steeting(human_steering_action):
    # this function convert the continuous steering action to discrete steering action and return the discrete action and the action space
    human_steering_action = round(human_steering_action,1)

    if -0.05 <= human_steering_action <= 0.05:
        action = 0
        action_space = 0
    if -0.2 <= human_steering_action < -0.05:
        action = -0.2  # LEFT_LEVEL_1
        action_space = 1
    if -0.4 <= human_steering_action < -0.2:
        action = -0.4  # LEFT_LEVEL_2
        action_space = 2
    if -0.6 <= human_steering_action < -0.4:
        action = -0.6  # LEFT_LEVEL_3
        action_space = 3
    if -0.8 <= human_steering_action < -0.6:
        action = -0.8  # LEFT_LEVEL_4
        action_space = 4
    if -1 <= human_steering_action < -0.8:
        action = -1  # LEFT_LEVEL_5
        action_space = 5
    if 0.05 < human_steering_action <= 0.2:
        action = 0.2  # RIGHT_LEVEL_1
        action_space = 6
    if 0.2 < human_steering_action <= 0.4:
        action = 0.4  # RIGHT_LEVEL_2
        action_space = 7
    if 0.4 < human_steering_action <= 0.6:
        action = 0.6  # RIGHT_LEVEL_3
        action_space = 8
    if 0.6 < human_steering_action <= 0.8:
        action = 0.8  # RIGHT_LEVEL_4
        action_space = 9
    if 0.8 < human_steering_action <= 1:
        action = 1  # RIGHT_LEVEL_5
        action_space = 10
        
    return action, action_space

# max_size_q = 30
# moving_average_q = Queue(maxsize=max_size_q)

def exp_moving_average(value, pre_value):
    alpha = 0.3
    if pre_value == 0:
        return value
    else:
        return alpha * value + (1 - alpha) * pre_value

    
def weighted_moving_average(values, period_num=None):
    
    if len(values) == 1:
        return values[0]
    
    # Ensure number of periods not exceeding size of values    
    if period_num == None:
        if len(values) < 8:
            period_num = len(values)
        else:
            period_num = 8


    weights_lst = [w for w in range(period_num,0,-1)]
    values_lst = [values[i] for i in range(-1,-period_num-1,-1)]
    
    weighted_lst = [weights_lst[i]*values_lst[i] for i in range(period_num)]
    
    wma = sum(weighted_lst) / sum(weights_lst)
    
    return wma
    

def mixing():
    # smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    pass

def get_feedback(human_st_action, opt_action):
    """If model is "RL": opt_action is the optimal action selected by the RL model,
    WHICH could be agent aciton of HUMAN action
    
    Note: At opt_action == HUMAN actions: no feedback need"""

    print("\nHuman Action:", human_st_action)
    print("Agent Action:", opt_action,"\n")

    human_steering = human_st_action
    opt_steering = opt_action

    diff = human_steering - opt_steering
    
    # if moving_average_q.full():
    #     moving_average_q.get()
    # else:
    #     moving_average_q.put(diff)
    # feedback_average = np.mean(moving_average_q.queue)
    # print("feedback_average",feedback_average)


    #if diff is not zero
    max_feedback = 130 # =ve turns the wheel CCW
    # if diff > 0.2:  
    #     if human_steering == 0 and opt_steering == 0:
    #         return 0
    #     if opt_steering == 0:
    #         return map_val(diff, 0, 2, 0, max_feedback) * -int(human_steering / abs(human_steering))
    #     elif human_steering == 0:
    #         return map_val(diff, 0, 2, 0, max_feedback) * int(opt_steering / abs(opt_steering))
    #     else:
    #         return map_val(diff, 0, 2, 0, max_feedback) * int(opt_steering / abs(opt_steering))
    # else:
    #     return 0
    print("Difference in steering:", diff)
    if diff < 0: # Haptic Force is CW
        return -1 * map_val(abs(diff), 0, 2, 0, max_feedback)
    elif diff > 0: # Haptics Force is CCW
        return map_val(abs(diff), 0, 2, 0, max_feedback)
    else:
        return 0



def map_val(input, input_min, input_max, output_min, output_max):
    input_span = input_max - input_min
    output_span = output_max - output_min

    # Convert the left range into a 0-1 range (float)
    valueScaled = (input - input_min) / float(input_span)

    # Convert the 0-1 range into a value in the right range.
    output = output_min + (valueScaled * output_span)
    return output

def main(
    alpha_schedule: List,
    total_timesteps: int,
    methods_schedule: List,
    feedback: bool,
    user_name: str,
    trial: int,
):
    
    feedback_value = 0
    frames_per_state = 4
    # best_model_9_dec
    # copilot_training_with_2M_0.85_init_eps_best_model
    model = DQNCopilot.load("/home/mohamed/catkin_ws/src/haptic-shared-autonomy-ros-pkg/scripts/copilot_training_with_2M_0.85_init_eps_best_model", device='cpu')

    abs_timestep = 0
    for method in methods_schedule:
        for feedback in feedback:
            # Releases all unoccupied cached memory currently held by the caching allocator
            th.cuda.empty_cache()

            # Creating evaluation databases
            results = {}
            results_df = pd.DataFrame(columns=["Method", "Alpha", "Score"])
            feedback_recorder_df = pd.DataFrame(columns=["feedback1", "feedback2", "human_action", "model_action","diff","alpha"])
            action_df = pd.DataFrame(columns=["human_percent", "agent_percent"])

            for alpha in alpha_schedule:
                # Initialize parameters
                actions_alpha_var_human = 0
                actions_alpha_var_agent = 0
                score = 0
                feedback_lst = []

                env = CarRacing(
                    allow_reverse=False,
                    grayscale=1,
                    show_info_panel=1,
                    discretize_actions="smooth_steering",  # n_actions = 11
                    num_tracks=2,
                    num_lanes=2,
                    num_lanes_changes=4,
                    max_time_out=5,
                    frames_per_state=frames_per_state,
                    display=f"Method {method} - Alpha = {alpha}",
                )

                env = wrappers.Monitor(
                    env,
                    f"./data_collected_{trial}/{user_name}/feedback_{feedback}/{method}/video/alpha_{alpha}/",
                    force=True,
                )
                observation = env.reset()

                # ROS Intialization
                rospy.init_node("car_control_node")
                feedback_pub = rospy.Publisher("/feedback", Int16, queue_size=10)
                score_pub = rospy.Publisher("/score", Float32, queue_size=10)
                
                done = False
                timestep = 0
                human_counter = 0
                agent_counter = 0

                # Observation space: 4 sequence image layers + steer action layer (copilot_obs)
                observation_space = spaces.Box(
                    low=0,
                    high=255,
                    shape=(STATE_H, STATE_W, frames_per_state + 1),
                    dtype=np.uint8,
                )
                copilot_obs = np.zeros(observation_space.shape)

                if rospy.is_shutdown():
                    feedback_pub.publish(0)
                    
                while not done:
                    msg = rospy.wait_for_message("/counter", Int16)
                    
                    # Set home position for steering wheel at timestep = 0
                    if abs_timestep == 0:
                        zero_counter = msg.data

                    # Bounding steering wheel angles between [-1, 1]
                    human_steering_action = min((msg.data - zero_counter) / 1100, 1) # upper limits = 1
                    human_steering_action = max(human_steering_action, -1) # lower limit = -1
                    print("human_steering_action",human_steering_action)

                    # Converting continous action to discrete
                    disc_human_steering_action, pi_action = cont2disc_steeting(human_steering_action)
                    print("disc_human_steering_action",disc_human_steering_action)
                    print("pi_action",pi_action)                    

                    # Creating the steering action layer
                    pi_action_steering_mapped = int(map_val(disc_human_steering_action, -1, 1, 0, 255))
                    pi_action_steering_frame = (
                    np.zeros((copilot_obs.shape[0], copilot_obs.shape[1]), dtype=np.int16)
                    + pi_action_steering_mapped)
                    
                    copilot_obs[:, :, 0:4] = observation  # Copy the first four channels from observation
                    copilot_obs[:, :, 4] = pi_action_steering_frame  # Assign pi_action_steering_frame to the fifth channel
                    
                    #cast copilot_obs to int
                    copilot_obs = copilot_obs.astype(np.uint8)
                    # print("copilot_loop", copilot_obs)

                    # Append the new frame tensor to the observation tensor along the first axis
                    # copilot_obs = th.cat((observation, new_frame), dim=0)

                    # Now 'new_observation' contains the original frames with the 5th frame appended

                    opt_action, _ = model.predict(copilot_obs)
                    print("opt_action",opt_action)

                    if method == "RL":
                        # Assuming 'observation' is a numpy array with shape (96, 96, 4)
                        # Transpose the array to match the expected input format (batch_size, channels, height, width)
                        # copilot_obs = copilot_obs.transpose((0, 3, 1,2))
                        # print("copilot_obs_55",copilot_obs.shape)

                        # Convert the numpy array to a PyTorch tensor
                        copilot_obs_tensor = th.tensor(copilot_obs, dtype=th.float32).to(
                            "cpu"
                        )
                        # Add batch dimension
                        copilot_obs_tensor = copilot_obs_tensor.unsqueeze(0)
                        copilot_obs_tensor = copilot_obs_tensor.permute(0, 3, 1,2)
                        # Pass the copilot_obs tensor to the model
                        q_values = model.policy.q_net.forward(copilot_obs_tensor)
                        q_values -= th.min(q_values)
                        # print("q_values",q_values[0])

                        pi_action_q_value = q_values[0][pi_action]
                        opt_action_q_value = q_values[0][opt_action]

                        if pi_action_q_value >= (1 - alpha) * opt_action_q_value:
                            action = pi_action
                            # print("human")
                            human_counter += 1
                            actions_alpha_var_human += 1
                            action_df = pd.concat([action_df, pd.DataFrame({"human_percent": 1, "agent_percent": 0}, index=[0])], axis=0)
                            # append(  
                            # {"human_percent": 1, "agent_percent": 0}, ignore_index=True)
                        else:
                            action = opt_action                          
                            # print("agent")
                            agent_counter += 1
                            actions_alpha_var_agent += 1
                            action_df = pd.concat([action_df, pd.DataFrame({"human_percent": 0, "agent_percent": 1}, index=[0])], axis=0)
                            # append(
                            # {"human_percent": 0, "agent_percent": 1}, ignore_index=True)

                      
                    elif method == "PIM":
                        agent_steering_action = disc2cont(opt_action)[0]
                        action_steering = (
                            alpha * human_steering_action
                            + (1 - alpha) * agent_steering_action
                        )

                        actions_alpha_var_human += 1
                        actions_alpha_var_agent += 1

                        action = disc2cont(opt_action)
                        action[0] = action_steering


                    # print("Applied Action", action)
                    observation_, reward, done, info = env.step(action)
                    env.render()
                    score += reward
                    observation = observation_

                    # Adding Haptic Feedback
                    if True:
                        model_action = disc2cont(action)[0] # DEFAULT WAS opt_action
                        prevous_feedback_value = feedback_value
                        feedback_value = int(get_feedback(human_steering_action, model_action))
                        
                        # Exponential Moving Average Smoothing
                        feedback_value1 = exp_moving_average(feedback_value, prevous_feedback_value)
                        prevous_feedback_value = feedback_value1
                        
                        # Weighted Moving Average Smoothing
                        feedback_lst = feedback_recorder_df["feedback2"].values.tolist()
                        feedback_lst.append(feedback_value)
                        feedback_value2 = weighted_moving_average(feedback_lst)
                        
                        print("feedback_value",feedback_value)
                        
                        #save all feedback values in a csv file with respect to the disc_human_steering_action and opt_action
                        feedback_recorder_df = pd.concat(
                            [feedback_recorder_df, pd.DataFrame({
                                "feedback1": feedback_value1, 
                                "feedback2": feedback_value2, 
                                "human_action": human_steering_action,
                                "model_action": model_action,
                                "alpha":alpha, 
                                "diff":abs(model_action-human_steering_action)}, index=[0])], axis=0
                        )
                        # append(
                        #     {"feedback": feedback_value, "human_action": disc_human_steering_action, 
                        #      "model_action": model_action,"alpha":alpha, "diff":abs(model_action-disc_human_steering_action)}, ignore_index=True
                        # )
                        
                        # Publish feedback value to arduino
                        if feedback:
                            feedback_pub.publish(int(feedback_value2))
                        else:
                            feedback_pub.publish(0)

                    # publish score value 
                    score_pub.publish(score)

                    print("timestep is", timestep, "\n")

                    if done and (timestep < total_timesteps):
                        env.reset()
                        done = False

                    if timestep >= total_timesteps:
                        if feedback:                            
                            feedback_pub.publish(0)
                        break


                    timestep += 1
                    abs_timestep += 1
                # print("human percent", human_counter / total_timesteps)
                # print("agent percent", agent_counter / total_timesteps)

                env.close()
                results[f"{method}_alpha_{alpha}"] = score
                results_df = results_df.append(
                    {"Method": method, "Alpha": alpha, "Score": score}, ignore_index=True
                )

                # Generate evaluation charts
                pie_chart_categories = ["human", "agent"]
                pie_chart_values = [actions_alpha_var_human, actions_alpha_var_agent]
                percentage_human = round((actions_alpha_var_human / (actions_alpha_var_human + actions_alpha_var_agent)*100),2)
                percentage_agent = round((actions_alpha_var_agent / (actions_alpha_var_human + actions_alpha_var_agent)*100),2)
                plt.figure()
                plt.pie(pie_chart_values, labels=pie_chart_categories)
                plt.title(f"actions per alpha {alpha}")
                plt.annotate("human= {} %".format(percentage_human), xy=(0.1, 1), xytext=(0.7, -1.2), color = "black")
                plt.annotate("agent= {} %".format(percentage_agent), xy=(0.1, 1), xytext=(0.7,-1.4), color = "black")
                plt.legend(loc="upper left")
                plt.savefig(
                    f"./data_collected_{trial}/{user_name}/feedback_{feedback}/{method}/pie_chart_for_actions_of_alpha_{alpha}.png"
                )
            feedback_recorder_df = pd.concat([feedback_recorder_df, action_df], axis=1)
            feedback_recorder_df.to_csv(
                f"./data_collected_{trial}/{user_name}/feedback_{feedback}/{method}/feedback_recorder.csv", index=False
            )

            pilots = list(results.keys())
            rewards = list(results.values())
            results = {}

            # plotting a line plot feedback and human_percent vs timestep
            figure, axis = plt.subplots(2, 2)
            axis[0,0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["feedback1"] ,label="feedback", color="red")
            axis[0,1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["diff"] ,label="diff", color="black")
            axis[1,0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["human_action"] ,label="human_action",color="green")
            axis[1,1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["model_action"] ,label="model_action",color="blue")
            figure.set_size_inches(40.5, 15.5)
            plt.savefig(
                f"./data_collected_{trial}/{user_name}/feedback_{feedback}/{method}/feedback_and_human_percent_vs_timestep.png"
            )

            # feedback_and_diff_vs_timestep.png
            figure_1, axis_1 = plt.subplots(2, 1)
            axis_1[0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["feedback2"] ,label="feedback_weighted", color="green")
            axis_1[0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["feedback1"] ,label="feedback_exp", color="red")
            axis_1[1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["diff"] ,label="diff", color="black")
            axis_1[0].set_title("feedback")
            axis_1[1].set_title("diff")
            figure_1.set_size_inches(40.5, 15.5)
            plt.savefig(
                f"./data_collected_{trial}/{user_name}/feedback_{feedback}/{method}/feedback_and_diff_vs_timestep_{alpha}.png"
            )

            figure_2, axis_2 = plt.subplots(2, 1)
            axis_2[0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["human_action"] ,label="human_action",color="green")
            axis_2[1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["model_action"] ,label="model_action",color="blue")
            axis_2[0].set_title("human_action")
            axis_2[1].set_title("model_action")
            figure_2.set_size_inches(40.5, 15.5)
            plt.savefig(
                f"./data_collected_{trial}/{user_name}/feedback_{feedback}/{method}/human_percent_vs_timestep.png"
            )

            # creating the bar plot
            plt.figure()
            plt.bar(pilots, rewards, color="blue", width=0.4)
            plt.xlabel("methods")
            plt.ylabel(f"score per {total_timesteps} timesteps")
            plt.title(f"various setups scores in {total_timesteps} timesteps")
            # plt.show()
            # plt.close()
            plt.savefig(
                f"./data_collected_{trial}/{user_name}/feedback_{feedback}/{method}/bar_chart_for_scores_of_alpha_{alpha}.png"
            )

            results_df.to_csv(
                f"./data_collected_{trial}/{user_name}/feedback_{feedback}/results.csv", index=False
            )
            review= input('Did the Co-Pilot Help you? (1/0))')
            with open(f'./data_collected_{trial}/feedback_{feedback}.csv', 'a') as f:
                f.write(f'{review}\n')

            # comment= input('which mode was the best? ')
            # with open(f'./data_collected_{trial}/select_mode_{feedback}.csv', 'a') as f:
            #     f.write(f'{comment}\n')


if __name__ == "__main__":

    alpha_schedule = [0.6] # percent of pilot (human) action
    total_timesteps = 900
    methods_schedule = ["RL"]
    feedback = [True]
    user_name = input("Enter your name: ")
    trial = 1
    
    main(alpha_schedule, total_timesteps, methods_schedule, feedback, user_name, trial) 

    # alpha_schedule_feedback = [0.6,0.8,1]   
    # feedback_y = [True]
    # main(alpha_schedule_feedback, total_timesteps, methods_schedule, feedback_y, user_name, trial)
