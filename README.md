# How to run code

## 1. Create Conda Environment:

**conda env create -f requirments.txt**

## 2. Activate Conda Environment:

**conda activate env**

## Discussion:

**Project Motivation**

This repo serves as a jumping off point for training low-parameter models to detect inappropriate actions in children's games. More specfically, interactions that occur in virtual space. As the proliferation of virtual reality (VR) technology continues to increase, there is a signifant concern that these technologies will become disproportionatly weaponized against at-risk populations, such as children. Therefore it is imperitive to counter act sexual harassment in children's VR games. Examples of this include, groping, sexually charged discussion from adults, and virtual rape. The focus of this exploratory project is to develop a submodle for the detection of virtual groping with a simple taxonomy that can be extended to low-parameters models to aid in this technologies adoption.


[Here is potentially how the bot could be used in near-real-time to evaluated interactions during game play](assets/Media1.mp4)


**Dataset Generation**

![Data collection was performed with a constant radius equal to 5meters in virtual space. The data collection bot performed a circular trajectory.](assets/bot_collection.png)
LLaVA models have shown great performance and generalization across many task. In fact, the LLaVA 1.6 34B parameter model showed incredible out-of-the-box performance on simple inferences about virtual environments, even though it was not explicitly trained primarily on this domain. In this work, we propose using the latest LLaVA 1.6 34B model to train a smaller model - LLaVA 1.5 7B. To do this, an image dataset was created in Unity3D. For this preliminary work, two interactions were depicted; 1.) Consensual interactions and 2.) Nonconsensual interaction. Then leveraging the simple taxonomy for determining virtual groping the LLaVA 1.6 was used to generate binary classifications of the images -a.) are they touching & b.) are they touching in an intimate area. These data was formatted as described by the original authors of the LLaVA models: [https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md]


![Data generation pipeline was created by leveraging the stronger LLaVA 1.6 34B parameter model to create question answer (QA) pairs. The QA pairs were used to fine-tune a LLaVA 1.5 7B parameter model - proposed to be the moderation bot.](assets/Parameter%20Efficient%20Moderation%20Bot.png)


**Results**

The results of the inital fine-tuning showed deterioration of performance and strong hallucenations. This was expected due to the number of data points in our training data (n=286). As we adopt more characters into the unity scene and increase the number of interactions depicted in the dataset, our hypothosis is there will be significant improvement in the performance of the lower parameter models for making inference on interactions in virtual space.
