import matplotlib.pyplot as plt

from tactix_env import TacTixEnv
from expectimax_agent import ExpectimaxTacTixAgent
from trainer_agent import TrainerAgent

def run_multiple_games(env, agent1, agent2, num_games=20):
    results = {
        "agent1_wins": 0,
        "agent2_wins": 0
    }

    for _ in range(num_games):
        obs = env.reset()
        done = False

        while not done:
            current_agent = agent1 if obs["current_player"] == 0 else agent2
            action = current_agent.act(obs)
            obs, reward, done, _ = env.step(action)

        last_player = 1 - obs["current_player"]

        if env.misere:
            winner = obs["current_player"]
        else:
            winner = last_player

        if winner == 0:
            results["agent1_wins"] += 1
        else:
            results["agent2_wins"] += 1

    return results

def plot_results(results):
    labels = ['Agent 1', 'Agent 2']
    counts = [results["agent1_wins"], results["agent2_wins"]]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=['skyblue', 'salmon'])
    plt.ylabel('Wins')
    plt.title('TacTix Tournament Results')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}', ha='center')

    plt.ylim(0, max(counts) + 10)
    plt.show()

if __name__ == "__main__":
    env = TacTixEnv(board_size=6)

    agent1 = ExpectimaxTacTixAgent(env, depth=5)
    agent2 = TrainerAgent(env)

    results = run_multiple_games(env, agent1, agent2, num_games=100)

    print("Resultados:", results)
    plot_results(results)
