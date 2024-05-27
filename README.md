CNN-based Imitation Learning for SuperTuxKart Ice-Hockey

William Chung
Furkan Goktas
Sameer Kausar
Shawn Tanner

## ABSTRACT

This paper presents the development and implementation of a state-based agent for playing the SuperTuxKart Ice hockey game. The agent is designed using Convolutional Neural Networks (CNN) and Imitation Learning; its aim is to score as many goals as possible in a 2 vs. 2 tournament against other agents developed by the course instructors. The methodology involves gathering and balancing training data, implementing a CNN model for real-time playing, training the network, and evaluating the agent's performance. The results demonstrate the effectiveness of the state-based approach in mastering complex game strategies without vision input. Future work includes exploring further enhancements to the agent's capabilities and performance, with specific attention to optimizing real-time decision-making.


Terms: SuperTuxKart, Deep Learning, Reinforcement Learning, Imitation Learning, Neural Networks, Convolutional Neural Networks, Pytorch.


## I. INTRODUCTION

SuperTuxKart is a popular open-source kart racing game, inspired by the classic Mario Kart series. Players control characters from various open-source games and projects, racing against each other on colorful tracks while using items to gain advantages or hinder opponents. The game is known for its engaging gameplay, vibrant graphics, and support for both single-player and multiplayer modes. This project focuses on developing a state-based agent for playing the SuperTuxKart ice hockey game, intending to explore the capabilities of deep learning in mastering this challenging game.

A. Aim

The aim of this project is to design and implement a state-based agent that can effectively play the SuperTuxKart ice hockey game, scoring as many goals as possible and winning matches in 2 vs. 2 tournaments against agents developed by the course instructors. By utilizing a state-based approach, we aim to leverage the information from each game state which includes data on player karts, opponent karts, and the ball to make informed decisions and execute effective strategies in real-time.

B. Objective

The primary objective of this project is to demonstrate the effectiveness of a state-based agent in playing the SuperTuxKart ice hockey game. This involves gathering and balancing training data, implementing a Convolutional Neural Network (CNN) model for real-time playing, and evaluating the agent's performance quantitatively and qualitatively. Additionally, we aim to explore the limitations of the state-based approach and identify potential areas for improvement and future research.

## II. BACKGROUND / RELATED WORK</strong> </p>


A. Machine Learning in Games

Machine learning (ML) techniques have found diverse applications in game development, including the development of non-player characters (NPCs) with realistic and adaptive behaviors. ML algorithms can train NPCs to exhibit human-like behaviors, such as learning from player interactions and adapting their strategies over time. ML is also used in procedural content generation (PCG), player analytics, personalization, game testing, and debugging. For our project, we will be leveraging machine learning techniques, specifically imitation and reinforcement learning, to develop a state-based agent for playing ice hockey in SuperTuxKart.

B. Deep Learning

Deep learning [1], a subfield of machine learning, has revolutionized the field of artificial intelligence. It involves the use of neural networks with multiple layers to learn complex patterns in data. Deep learning has achieved remarkable success in various domains, including computer vision, natural language processing, and game playing. In the context of our project, we will be using deep learning techniques, specifically convolutional neural networks (CNNs), to analyze game state data and make informed decisions.

C. Imitation Learning

Imitation learning [2] is a machine learning paradigm where an agent learns by imitating the behavior demonstrated by expert demonstrations. In the context of game-playing agents, imitation learning can be used to train agents to mimic the behavior of expert players, enabling them to perform well in complex game environments. For our project, we will explore the use of imitation learning techniques to train our agents using expert demonstrations from the provided agents developed by the course instructors.

D. Reinforcement Learning

Reinforcement learning [3] is another machine learning paradigm that has been widely used in the development of game-playing agents. In reinforcement learning, an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties. The agent's goal is to maximize the cumulative reward over time, leading to the emergence of intelligent behavior. In our project, we will be using reinforcement learning to train our agent to play SuperTuxKart ice hockey and to develop an agent that can effectively compete against human players.

E. SuperTuxKart

SuperTuxKart [4] is an open-source kart racing game, similar to Mario Kart, that offers a fun and engaging gaming experience. It features a variety of tracks, characters, and items, making it a popular choice among gamers. The game's open-source nature has led to a vibrant community of developers and modders, contributing to its ongoing development and popularity.

## III. DESIGN AND METHODOLOGY 

A. Gathering Training Data

To collect data for training our agent, we first selected the 'best' supplied test agent for imitation learning. This selection was made by running each test agent through the known test set and choosing the agent with the highest score. Jurgen was identified as the best-performing agent, achieving a score of 97 points with its default kart ‘Sarah_the_racer’.


<table>
  <tr>
   <td><strong>Agent</strong>
   </td>
   <td><strong>Score</strong>
   </td>
  </tr>
  <tr>
   <td>Jurgen
   </td>
   <td>97
   </td>
  </tr>
  <tr>
   <td>Image_jurgen
   </td>
   <td>67
   </td>
  </tr>
  <tr>
   <td>Yann
   </td>
   <td>49
   </td>
  </tr>
  <tr>
   <td>Goeffrey
   </td>
   <td>41
   </td>
  </tr>
  <tr>
   <td>Yoshua
   </td>
   <td>22
   </td>
  </tr>
</table>


**Figure 1:** Score when using supplied agents

We modified the supplied code to run a loop of round-robin matches between our agent (Jurgen) and randomly selected opponents. During each match, the team designation (0 or 1), puck starting position, and opponents (test agents) were randomly chosen from predefined distributions. The puck's starting position was determined based on a random choice of six distributions, ensuring variability in the game environment. 

The max score was set to one for each match, and only matches resulting in our agent scoring a goal were saved. The collected data included the state and action variables of each player from our team.  This data collection process continued until 445,000 frame records were captured, ensuring a sufficient amount of data for training our agent. Additionally, we planned to further enhance our agent using reinforcement learning (Temporal-Displacement Deep Q-Network (TDDQN)) if needed and if time permits.

B. Agent state variables

Variables to control the agent were chosen to be minimal while providing sufficient information to control the kart.  In the end, we chose seven variables, five of which were calculated from the base state variables, for input to the agent control network:
1. Kart x position
2. Kart y position
3. Puck to goal angle
4. Kart angle
5. Kart to pick angle
6. Kart to goal angle
7. Puck to goal distance magnitude

We included the ‘puck to goal distance magnitude’ parameter to allow for the possibility of different behaviors such as being more offensive when the kart is closer to the goal or being more defensive when it is farther away from the goal.  These variables are illustrated in Figure 2.
   <div style="text-align: center;">
    <img src="https://github.com/cualum/SuperTuxKart-Deep-Learning/assets/137105371/d37c7b35-4b8c-4847-b067-63f5a998fe57" width="400">
   </div>   
** Figure 2: Illustration of state variables for agent

C. Agent Control Network

Pytorch [5] was used for this project.  The agent control network takes an input tensor of 14 (seven states for each player).  This input is split into a group of seven for the first player and a group of seven for the second player.  The  two groups are passed two identical parallel subnetworks (one for each kart) with the action outputs  from each network concatenated as the network output.  This allows for the possibility of each kart learning different behaviors while maintaining a single network architecture.  Each parallel subnetwork consisted of three fully connected layers followed by ReLU activation functions.  The first layer maps the seven input features to a hidden layer of size 512.  This process is repeated with another hidden layer of size 512 and a final output layer of size three.  Each output is concatenated forming a final output of size 6.
   <div style="text-align: center;">
    <img src="https://github.com/cualum/SuperTuxKart-Deep-Learning/assets/137105371/ee841307-7d97-440d-91c0-205ff3c57f4b" width="400">
   </div>   
** Figure 3: Network architecture


D.  Training parameters

The training process utilized the Adam optimizer with a learning rate of 0.001 and a weight decay of 1e-5.  Using the MultiStepLR scheduler, the learning rate was adjusted every ten iterations starting at 20 with a gamma of 0.5.  The loss function was the L1Loss from the torch.nn library.  After some experimentation, a final batch size of 150 was used. A typical loss plot is shown in Figure 4.

E.  Overfitting

To avoid overfitting, two strategies were invoked; early stopping, and model complexity limitation.  Admittedly this aspect of the work was not as controlled and investigated as it should be, in part, due to time constraints.  Fitting was typically stopped around 50 epochs even though more epochs could produce lower loss.  Additionally, the model inputs were minimized (seven parameters per player) and the model size was kept to a minimum (three fully connected layers).  Given more time, this space could be fully investigated and likely better results obtained.  The metric for this agent was the number of goals scored against four other agents throughout eight games, converted to a score value. Ultimately we obtained a score of 91 on the known test set and a score of 88 on the unknown test set so even though our method was not well mapped, it produced a result that indicates good generalization.  
   <div style="text-align: center;">
    <img src="https://github.com/cualum/SuperTuxKart-Deep-Learning/assets/137105371/cdb94fe5-2ae5-4b24-b8f0-5bc9f973e267" width="400">
   </div>  
** Figure 4: Loss vs epochs

## IV. RESULTS
A score of 73 using the known test set was obtained using ‘Tux’ as our kart.  At this point, we experimented with changing the kart on this trained network.  The results are shown in Figure 5.  Each kart has four attributes in the game:  1) mass, 2) maximum speed, 3) acceleration, and 4) nitro efficiency.  Inspection of the in-game statistics for the karts shows there are basically three classes of karts.  Light karts have lower mass, lower top speed, and higher acceleration.  Heavy karts have a higher mass, higher top speed, and lower acceleration.  Medium karts are in the middle of these two.  Inspection of the karts in-game suggests that karts of a similar type are very similar in their attributes, at least as far as can be discerned from the in-game graphic.  The actual number could not be found.  Generally speaking, light karts did the best, followed by medium then heavy as shown by the local grader results.  It was somewhat surprising to see that although ‘Sarah_the_racer’ was used to obtain the training data, it was not the best-performing kart for our trained agent.  It was also surprising that ‘Kiki’, a light kart, performed so poorly.  At least visually, the properties/stats of ‘Kiki’ look almost identical to those of ‘Wilber’.  We can only conclude that the in-game graphic is misleading or missing information.  It was also somewhat surprising that there was such a large range of results between the karts.


<table>
  <tr>
   <td><strong>Kart</strong>
   </td>
   <td><strong>Kart Type</strong>
   </td>
   <td><strong>Score</strong>
   </td>
  </tr>
  <tr>
   <td>Wilber
   </td>
   <td>Light
   </td>
   <td>91
   </td>
  </tr>
  <tr>
   <td>Hexley
   </td>
   <td>Light
   </td>
   <td>76
   </td>
  </tr>
  <tr>
   <td>Tux
   </td>
   <td>Medium
   </td>
   <td>73
   </td>
  </tr>
  <tr>
   <td>Sarah_the_racer
   </td>
   <td>Light
   </td>
   <td>70
   </td>
  </tr>
  <tr>
   <td>Xue
   </td>
   <td>Medium
   </td>
   <td>69
   </td>
  </tr>
  <tr>
   <td>Gnu
   </td>
   <td>Medium
   </td>
   <td>69
   </td>
  </tr>
  <tr>
   <td>Konqi
   </td>
   <td>Medium
   </td>
   <td>66
   </td>
  </tr>
  <tr>
   <td>Beastie
   </td>
   <td>Heavy
   </td>
   <td>65
   </td>
  </tr>
  <tr>
   <td>Suzanne
   </td>
   <td>Medium
   </td>
   <td>63
   </td>
  </tr>
  <tr>
   <td>Gavroche
   </td>
   <td>Medium
   </td>
   <td>63
   </td>
  </tr>
  <tr>
   <td>Emule
   </td>
   <td>Medium
   </td>
   <td>60
   </td>
  </tr>
  <tr>
   <td>Pidgin
   </td>
   <td>Heavy
   </td>
   <td>50
   </td>
  </tr>
  <tr>
   <td>Kiki
   </td>
   <td>Light
   </td>
   <td>48
   </td>
  </tr>
  <tr>
   <td>Nolok
   </td>
   <td>Medium
   </td>
   <td>41
   </td>
  </tr>
  <tr>
   <td>Amanda
   </td>
   <td>Heavy
   </td>
   <td>41
   </td>
  </tr>
  <tr>
   <td>Adiumy
   </td>
   <td>Medium
   </td>
   <td>40
   </td>
  </tr>
  <tr>
   <td>Sarah_the_wizard
   </td>
   <td>Medium
   </td>
   <td>40
   </td>
  </tr>
  <tr>
   <td>Puffy
   </td>
   <td>Heavy
   </td>
   <td>28
   </td>
  </tr>
</table>
** Figure 5: Score with various karts


<table>
  <tr>
   <td>Agent
   </td>
   <td>Known test set (goals in 8 games)
   </td>
   <td>Unknown test set (goals in 4 games)
   </td>
  </tr>
  <tr>
   <td><strong>Geoffrey</strong>
   </td>
   <td>8 
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td><strong>Jurgen</strong>
   </td>
   <td>6
   </td>
   <td>5
   </td>
  </tr>
  <tr>
   <td><strong>Yann</strong>
   </td>
   <td>10 
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td><strong>Yoshua</strong>
   </td>
   <td>7
   </td>
   <td>5
   </td>
  </tr>
</table>


** Figure 6: Score vs each agent


## VII. CONCLUSION
Using simple imitation learning of the test agent Jurgen with the kart ‘Sarah_the_racer’, we were able to obtain a score of 91 on the known test set and a corresponding score of 88 on the unknown test set.  The results vs each opponent agent are shown in Figure 6.  We would rate the following in order as having the most impact on the success or failure of this project:



1. Quality data collection
2. State variable selection
3. DNN design

Data collection is where the overwhelming amount of time was spent.


        **VIII. FUTURE SCOPE OF WORK**

Two notable approaches for enhancing decision-making in dynamic game environments such as SuperTuxKart are Dataset Aggregation (DAGGER) and Temporal-Difference Deep Q-Network (TDDQN).

DAGGER, proposed by Ross et al. in 2011 [1], is a method aimed at iteratively refining an agent's policy by combining initial expert demonstrations with ongoing agent-enacted trajectories. This approach addresses deviations from expert behavior and allows for periodic reintroduction of expert input during training sessions, potentially leading to a more robust decision-making process. By iteratively correcting the agent's behavior, DAGGER effectively bridges the gap between the theoretical model and its practical effectiveness, adapting the agent more suitably to the dynamic game environment.

TDDQN, as proposed by Sutton et al. in 2018 [2], offers a way to enhance decision-making by considering temporal differences in state values. By evaluating actions not only for immediate rewards but also for long-term outcomes, TDDQN enables strategic planning in dynamic game environments. In games like SuperTuxKart, where long-term strategy significantly influences performance outcomes, TDDQN can help the agent make more forward-thinking decisions, leading to more nuanced and strategic behaviors.

Integrating DAGGER for iterative refinement and TDDQN for long-term strategic planning could lead to significant enhancements in agent performance in dynamic game environments like SuperTuxKart.

These improvements would greatly enhance the intelligence and effectiveness of our game-playing agents, pushing forward the capabilities of artificial intelligence in gaming and other complex environments.

**REFERENCES**

**      **



1. Goodfellow-et-al-2016, Deep Learning. MIT Press.  [http://www.deeplearningbook.org](http://www.deeplearningbook.org) (2016)
2. Zare, M., Kebria, P., Khosravi, A., Nahavandi, S. A survey of Imitation Learning: Algorithms, Recent Developments, and Challenges. arXiv:2309.02473v1 [cs.LG]
3. Sutton, R. Barto, A. Reinforcement Learning: An Introduction 2nd ed.  The MIT Press (2018)
4. [https://supertuxkart.net/Main_Page](https://supertuxkart.net/Main_Page)
5. https://pytorch.org/
6. Ross, S., Gordon, G. J., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 627-635).
7. Sutton, R. S., Mahmood, A. R., & White, M. (2018). Imagination-augmented agents for deep reinforcement learning. In Advances in neural information processing systems (pp. 5690-5701)
