# DQN_Temperature
 Project carried out to learn and improve the energy required to control a server using a DQN algorithm and neural network

The project is structured in three main components that interact with each other to develop and improve the ability of an artificial intelligence (AI) to make decisions in a simulated environment.

Environment: This component acts as the scenario or "playing field" for our AI. It defines the rules, goals and constraints within which the AI must operate. The environment presents different situations or states that the AI needs to evaluate and respond to.

DQN (Deep Q-Network) algorithm: This algorithm is the decision-making core of our system. It uses reinforcement learning techniques to evaluate the actions taken by the AI, determining how good those actions are in terms of achieving the goals set in the environment. The DQN is responsible for updating the AI's decision "policy" based on the feedback received after each executed action.

Brain: This component is effectively the neural network that learns and optimises its behaviour over time. Through training with the DQN algorithm, the brain learns to interpret the state of the environment and make informed decisions that maximise cumulative rewards. Over time, the brain becomes more adept at handling the complexities of the environment and achieving goals efficiently.