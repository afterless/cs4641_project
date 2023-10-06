# Introduction/Background

The objective of this project is to create a Machine Learning model that can predict the potential salary of a job position based on details provided within a LinkedIn job posting. The chosen dataset to be used for training contains around 15,000 LinkedIn posts collected in a period of two days. From those posts, information was extracted and organized in CSV format with 27 different features such as location, company name, number of applications, etc.  

There have been past efforts to build a model to predict salary from job posts using various ML methods. One of them is using a bidirectional-GRU-CNN model suggested by Wang, et al, which has been proved to have a higher accuracy than other traditional models [1]. Another effort focused on creating a regression model to predict stock prices using large language models (LLMs), utilizing textual information [2]. Various statistical ML methods have been utilized for salary predictions around the world [3]. 
# Problem Definition

LinkedIn, a leading job platform, offers extensive information about companies and job openings. However, many companies do not disclose salary details in their postings. This omission can lead to frustration for potential applicants, as they lack crucial information for decision-making. Access to salary data can streamline the application process, allowing candidates to make well-informed choices and develop effective negotiation strategies. 
# Methods
For our data preprocessing and analysis, we will employ a set of common Python libraries and packages to efficiently handle various tasks. These include NumPy, pandas, scikit-learn, and NLTK. These packages are specifically utilized for data science, ML, and NLP tasks. 
## Potential Project Ideas

- Supervised Learning: 
    - We aim to fine-tune pretrained transformer models using a few different types of textual data from LinkedIn job postings, such as the job descriptions, necessary skills, and job titles. We aim to train these as regression models to estimate salaries. We will explore pretrained large language models, like GPT and BERT, which are based on neural networks. We will rely on the Hugging Face library, which provides implementations of these models. We can also experiment with this as classification models, by creating salary-range tier classifications instead of predicting a specific salary. For the training process, we will utilize PyTorch.

- Unsupervised Learning: 
    - We will analyze the same types of textual data and use NLP methods like TF-IDF, and then perform K-Means clustering to group similar job postings into cohesive categories. We will use existing scikit-learn libraries here.

# Potential Results & Discussions

This project aims to analyze employer-provided job posting data to identify essential information for potential applicants, aiding their decision-making process. The primary objective is to predict salary based on factors like skills, role descriptions, and job titles. Model effectiveness will be assessed using scikit-learn metrics. This includes absolute error and F1 scores for salary-range classifications and $R^2$ scores for a regression fit, representing quantifiable feedback on model performance [4]. Feature importance scoring will highlight crucial parameters. These metrics will give a holistic yet quantitative performance evaluation, which will guide continuous model refinement. 
# Proposed Timeline
A link to our prposed timeline can be found [here](https://gtvault-my.sharepoint.com/:x:/g/personal/achennak3_gatech_edu/EQ1QSjaV_FNAkyNbvHxFvkIBXh1E5bZUPvVsOEeYK_luOQ?e=4OJZDS)
# Contribution Table

| Task                             | Contributing Team Members(s)    |
|:---------------------------------|--------------------------------:|
| Research Potential Datasets      | Akul, Alex, Bao, Nikhil         |
| Discuss Topics/Ideas for Project | Akul, Alex, Ayush, Bao, Nikhil  |
| Find Sources                     | Akul, Bao, Nikhil               |
| Introduction & Background        | Bao                             |
| Problem Defintion                | Bao                             |
| Methods                          | Akul                            |
| Potential Results & Discussion   | Nikhil                          |
| Proposed Timeline (Gantt Chart)  | Alex                            |
| Contribution Table & Checkpoint  | Akul                            |
| Compile References               | Akul, Nikhil                    |
| Create Presentation              | Akul, Bao, Nikhil               |
| Record/Edit/Upload Presentation  | Akul, Nikhil                    |
| Github Page Creation/Management  | Ayush                           |

# Checkpoint
Here is the link to our [dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

By the midterm report, we aim to have our dataset fully cleaned and preprocessed. In addition, we will have created and tested a model to see how well we are able to predict salaries from the scoring metrics we had mentioned previously. At this point, we will know if it will be worth continuing testing regression models, or if it would make more sense to pivot to a classification model. We will tune hyperparameters and see how this affects the performance of models. By the final report, we aim to have selected the best performing model out of all our experiments. We will plot the performances of different models to compare metrics through data visualization. By varying hyperparameters and specific features used for the training process, we hope to see what works best and optimizes the performance of our model. 
# References
[1] Z. Wang, S. Sugaya, and D. P. T. Nguyen, “[PDF] Salary Prediction using Bidirectional-GRU-CNN Model,” Association for Natural Language Processing, Mar. 2019. 

[2] P. Sonkiya, V. Bajpai, and A. Bansal, “Stock price prediction using BERT and GAN,” arXiv.org, Jul. 18, 2021. https://arxiv.org/abs/2107.09055 (accessed Oct. 06, 2023). 

[3] Y. T. Matbouli and S. M. Alghamdi, “Statistical Machine Learning Regression Models for Salary Prediction Featuring Economy Wide Activities and Occupations,” Information, vol. 13, no. 10, p. 495, Oct. 2022, doi: 10.3390/info13100495. 

[4] X. Geerinck, “Artificial Intelligence — How to measure performance — Accuracy, Precision, Recall, F1, ROC, RMSE, F-Test and R-Squared,” Medium, Jan. 03, 2020. Accessed: Oct. 06, 2023. [Online]. Available: https://medium.com/@xaviergeerinck/artificial-intelligence-how-to-measure-performance-accuracy-precision-recall-f1-roc-rmse-611d10e4caac 
