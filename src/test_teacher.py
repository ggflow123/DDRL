import gym
from teacher import PPOTeacherCartpole
from stable_baselines3.common.evaluation import evaluate_policy




def evaluate_teacher(teacher, env, num_episodes=100):
    """
    Evaluate the teacher over a number of episodes.

    :param teacher: The Teacher object to evaluate.
    :param env: The gymnasium environment to use for evaluation.
    :param num_episodes: Number of episodes to run for the evaluation.
    :return: The average reward over the number of episodes.
    """
    total_rewards = 0.0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        state = state[0]

        while not done:
            action = teacher.step(state)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward

        total_rewards += episode_reward

    average_reward = total_rewards / num_episodes
    return average_reward

def main():
    env = gym.make('CartPole-v1')

    #log = "./log"

    teacher = PPOTeacherCartpole(model_name="PPO_Teacher", env=env)
    #teacher = download_from_hub(algo='ppo', env_name='CartPole-v1', exp_id=0, folder=log, organization="sb3", force=False)

    #teacher.train(total_timesteps=10000)

    mean_reward, std_reward = evaluate_policy(
        teacher.model, env, render=False, n_eval_episodes=5, deterministic=True, warn=False
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Assuming you have already created and trained a Teacher instance
    # average_reward = evaluate_teacher(teacher, env, num_episodes=100)
    # print(f"Average Reward: {average_reward}")


if __name__ == "__main__":
    main()