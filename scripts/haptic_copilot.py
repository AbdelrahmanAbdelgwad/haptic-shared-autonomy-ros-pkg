import rospy
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

STATE_H = 96
STATE_W = 96

#convert discrete action to continuous action
def disc2cont(action):
    if action == 0:
        action = [0, 0.4, 0.1]  # "NOTHING"
    elif 1 <= action <= 5:
        action = [round((-0.2*action),1), 0.4, 0.05]
    elif 6 <= action <= 10:
        action = [round(0.2*(action-5),1), 0.4, 0.05]
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


def get_feedback(human_st_action, opt_action):

    human_steering = human_st_action
    opt_steering = opt_action
    
    diff = abs(human_steering - opt_steering)
    
    #if diff is not zero
    if diff > 0.2:  
        if opt_action == 0:
            return map_val(diff, 0, 2, 0, 60) * -int(human_steering / abs(human_steering))
        elif human_steering == 0:
            return map_val(diff, 0, 2, 0, 60) * int(opt_steering / abs(opt_steering))
        else:
            return map_val(diff, 0, 2, 0, 60)* (int(opt_steering / abs(opt_steering)))      
    else:
        return 0



def map_val(input, input_min,input_max, output_min, output_max):
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
    results = {}
    results_df = pd.DataFrame(columns=["Method", "Alpha", "Score"])
    feedback_recorder_df = pd.DataFrame(columns=["feedback", "human_action", "model_action","diff","alpha"])
    action_df = pd.DataFrame(columns=["human_percent", "agent_percent"])
    frames_per_state = 4
    model = DQNCopilot.load("best_model_dec", device='cuda')

    abs_timestep = 0
    for method in methods_schedule:
        for alpha in alpha_schedule:
            actions_alpha_var_human = 0
            actions_alpha_var_agent = 0
            score = 0
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
                f"./data_collected_{trial}/{user_name}/feedback/{method}/video/alpha_{alpha}/",
                force=True,
            )
            observation = env.reset()

            rospy.init_node("car_control_node")
            feedback_pub = rospy.Publisher("/feedback", Int16, queue_size=10)
            score_pub = rospy.Publisher("/score", Float32, queue_size=10)
            done = False
            timestep = 0
            human_counter = 0
            agent_counter = 0
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
                # print("copilot_obs",copilot_obs.shape)
                if abs_timestep == 0:
                    zero_counter = msg.data
                human_steering_action = min((msg.data - zero_counter) / 1000, 1)
                human_steering_action = max(human_steering_action, -1)
                print("human_steering_action",human_steering_action)
                disc_human_steering_action, pi_action = cont2disc_steeting(human_steering_action)
                print("disc_human_steering_action",disc_human_steering_action)
                print("pi_action",pi_action)
                # pi_action = cont2disc(human_steering_action)

                pi_action_steering_mapped = int(map_val(disc_human_steering_action, -1, 1, 0, 255))

                pi_action_steering_frame = (
                np.zeros((copilot_obs.shape[0], copilot_obs.shape[1]), dtype=np.int16)
                + pi_action_steering_mapped)
                # print("pi_action_steering_frame",pi_action_steering_frame)
                
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
                        "cuda"
                    )
                    # Add batch dimension
                    copilot_obs_tensor = copilot_obs_tensor.unsqueeze(0)
                    copilot_obs_tensor = copilot_obs_tensor.permute(0, 3, 1,2)
                    # Pass the copilot_obs tensor to the model
                    q_values = model.policy.q_net.forward(copilot_obs_tensor)
                    q_values -= th.min(q_values)
                    print("q_values",q_values[0])

                    pi_action_q_value = q_values[0][pi_action]
                    opt_action_q_value = q_values[0][opt_action]

                    if pi_action_q_value >= (1 - alpha) * opt_action_q_value:
                        action = pi_action
                        print("human")
                        human_counter += 1
                        actions_alpha_var_human += 1
                        action_df = pd.concat([action_df, pd.DataFrame({"human_percent": 1, "agent_percent": 0}, index=[0])], axis=0)
                        # append(  
                        # {"human_percent": 1, "agent_percent": 0}, ignore_index=True)
                    else:
                        action = opt_action
                        print("agent")
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
                    action = disc2cont(opt_action)
                    action[0] = action_steering
                observation_, reward, done, info = env.step(action)
                env.render()
                score += reward
                observation = observation_

                if feedback:
                    model_action = disc2cont(opt_action)[0]
                    feedback_value = int (get_feedback(disc_human_steering_action, model_action))
                    #save all feedback values in a csv file with respect to the disc_human_steering_action and opt_action
                    feedback_recorder_df = pd.concat(
                        [feedback_recorder_df, pd.DataFrame({"feedback": feedback_value, "human_action": disc_human_steering_action, 
                         "model_action": model_action,"alpha":alpha, "diff":abs(model_action-disc_human_steering_action)}, index=[0])], axis=0
                    )
                    # append(
                    #     {"feedback": feedback_value, "human_action": disc_human_steering_action, 
                    #      "model_action": model_action,"alpha":alpha, "diff":abs(model_action-disc_human_steering_action)}, ignore_index=True
                    # )
                    
                    print("feedback_value",feedback_value)
                    feedback_pub.publish(feedback_value)

                score_pub.publish(score)

                print("timestep is", timestep, "\n")
                if done and (timestep < total_timesteps):
                    env.reset()
                    done = False

                if timestep >= total_timesteps:
                    if feedback:
                        feedback_value = 0
                        feedback_pub.publish(feedback_value)
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
                f"./data_collected_{trial}/{user_name}/feedback/{method}/pie_chart_for_actions_of_alpha_{alpha}.png"
            )
        feedback_recorder_df = pd.concat([feedback_recorder_df, action_df], axis=1)
        feedback_recorder_df.to_csv(
            f"./data_collected_{trial}/{user_name}/feedback/{method}/feedback_recorder.csv", index=False
        )


        pilots = list(results.keys())
        rewards = list(results.values())
        results = {}

        # plotting a line plot feedback and human_percent vs timestep
        figure, axis = plt.subplots(2, 2)
        axis[0,0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["feedback"] ,label="feedback", color="red")
        axis[0,1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["diff"] ,label="diff", color="black")
        axis[1,0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["human_action"] ,label="human_action",color="green")
        axis[1,1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["model_action"] ,label="model_action",color="blue")
        figure.set_size_inches(40.5, 15.5)
        plt.savefig(
            f"./data_collected_{trial}/{user_name}/feedback/{method}/feedback_and_human_percent_vs_timestep.png"
        )

        figure_1, axis_1 = plt.subplots(2, 1)
        axis_1[0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["feedback"] ,label="feedback", color="red")
        axis_1[1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["diff"] ,label="diff", color="black")
        axis_1[0].set_title("feedback")
        axis_1[1].set_title("diff")
        figure_1.set_size_inches(40.5, 15.5)
        plt.savefig(
            f"./data_collected_{trial}/{user_name}/feedback/{method}/feedback_and_diff_vs_timestep.png"
        )

        figure_2, axis_2 = plt.subplots(2, 1)
        axis_2[0].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["human_action"] ,label="human_action",color="green")
        axis_2[1].plot(np.arange(0,len(feedback_recorder_df)),feedback_recorder_df["model_action"] ,label="model_action",color="blue")
        axis_2[0].set_title("human_action")
        axis_2[1].set_title("model_action")
        figure_2.set_size_inches(40.5, 15.5)
        plt.savefig(
            f"./data_collected_{trial}/{user_name}/feedback/{method}/human_percent_vs_timestep.png"
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
            f"./data_collected_{trial}/{user_name}/feedback/{method}/bar_chart_for_scores_of_alpha_{alpha}.png"
        )

    results_df.to_csv(
        f"./data_collected_{trial}/{user_name}/feedback/results.csv", index=False
    )



if __name__ == "__main__":
    methods_schedule = ["RL"]
    alpha_schedule = [0.4, 0.6, 0.8, 1]   
    total_timesteps = 600
    feedback = True
    user_name = "anas_trial_best_model_Nov_8_nosisy"
    trial = 1
    main(alpha_schedule, total_timesteps, methods_schedule, feedback, user_name, trial)
