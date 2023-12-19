import numpy as np
import connection as cn

socket_connection = cn.connect(2037)

REWARD_FALL = -100
REWARD_WIN = 300
num_states = 96
num_actions = 3  # left, right, jump

q_table = np.zeros((num_states, num_actions))


alpha = 0.1  # taxa de aprendizado
gamma = 0.9  # fator de desconto
epsilon = 0.1  # epsilon-greedy
num_episodes = 500

# Função para extrair informações do estado binário
def extract_state_info(binary_state):
    platform = int(binary_state[:-2], 2)  # plataforma
    direction = int(binary_state[-2:], 2)  # direção
    return platform, direction

# Função para mapear plataforma e direção para um estado único
def map_to_single_state(platform, direction):
    return (platform * 4) + direction

def check_episode_done(reward):
    if reward == REWARD_FALL or reward == REWARD_WIN: return True
    else: return False

# Loop de treinamento
for episode in range(num_episodes):
    done = False
    current_state = 0
    
    while not done:
        # escolha por epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)  # ação aleatória
        else:
            action = np.argmax(q_table[current_state])  # melhor ação conhecida
        
        # execução da ação
        actions_mapping = {0: "left", 1: "right", 2: "jump"}
        action_name = actions_mapping[action]
        next_state, reward = cn.get_state_reward(socket_connection, action_name)
        
        platform, direction = extract_state_info(next_state)
        next_state = map_to_single_state(platform, direction)
        
        # atualização da q-table
        best_next_action = np.argmax(q_table[next_state])
        q_table[current_state, action] = (1 - alpha) * q_table[current_state, action] + alpha * (reward + gamma * q_table[next_state, best_next_action])
        
        # salvar q-table como .txt
        txt = open('q_table.txt', 'w')
        for line in range(num_states):
            txt.write(f"{q_table[line][0]:.5f} {q_table[line][1]:.5f} {q_table[line][2]:.5f}\n")
        txt.close()

        if check_episode_done(reward):
            done = True
        else:
            current_state = next_state
