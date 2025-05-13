<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI%20Gym-000000?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-blue?style=for-the-badge"/>
</p>

---

# 🎮 Policy Gradient Reinforcement Learning (REINFORCE)

A minimal yet powerful TensorFlow implementation of the **REINFORCE algorithm** (Monte Carlo Policy Gradient) trained on the classic control environment **CartPole-v1** from OpenAI Gym.

This project is part of a hands-on reinforcement learning exploration for educational purposes and model-sharing on the Hugging Face Hub.

---

## 🚀 Features

- ✅ Simple and intuitive TensorFlow-based REINFORCE agent  
- 📊 Trains on `CartPole-v1` using episodic return-based learning  
- 📈 Tracks and prints average rewards every 100 episodes  
- 🤗 Hugging Face Hub integration (optional)  
- 🧪 Includes agent evaluation after training  

---

## 📁 Project Structure
Policy-Gradient-RL/
├── code.py             # Full REINFORCE implementation
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and documentation

---

## 📦 Installation

Install dependencies using `pip`:

```bash
pip install numpy gym tensorflow matplotlib huggingface_hub imageio
```
Make sure you’re using Python 3.10 or above.

## 🧠 How It Works
A neural network policy is trained to map states to action probabilities.
- Actions are sampled stochastically from the policy.
- Returns are computed and standardized for each episode.
- Policy gradients are computed using TensorFlow's GradientTape.

## Run the training with:
```bash
python code.py
```
## 🧪 Evaluation
After training, the agent is evaluated using:
```python
evaluate_agent(eval_env, max_t, n_evaluation_episodes, policy)
```
This function reports the **mean** and **standard deviation** of total rewards across evaluation episodes.

## 📈 Sample Output
Example training output:
```yaml
Episode 100   Average Score: 37.56
Episode 200   Average Score: 71.23
...
Episode 3000  Average Score: 499.88
```
The agent gradually learns to balance the pole almost perfectly.

## 📌 To-Do

- [x] Build REINFORCE agent in TensorFlow  
- [ ] Add entropy regularization  
- [ ] Upload trained model to Hugging Face 🤗  
- [ ] Support continuous action spaces  
- [ ] Log metrics to TensorBoard  
