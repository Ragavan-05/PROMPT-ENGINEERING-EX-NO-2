# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:

Accuracy

Coherence

Simplicity

Speed

User experience

## OUTPUT:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/content/titanic_dataset (1).csv")  # Replace with actual path

# Display basic info
df.info()

# Display first few rows
df.head()

# Check shape
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

# Set PassengerId as index
df.set_index("PassengerId", inplace=True)

# Summary statistics
df.describe()

# Count unique values in categorical columns
categorical_columns = ["Survived", "Pclass", "Sex", "Embarked"]
for col in categorical_columns:
    print(f"{col} unique values:\n", df[col].value_counts(), "\n")

sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.show()

df["Pclass"].unique()
df.rename(columns={"Sex": "Gender"}, inplace=True)
df

sns.catplot(x='Survived', hue='Gender', data=df, kind='count')
plt.title("Survival by Gender")
plt.show()

sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age Distribution by Survival")
plt.show()

sns.boxplot(x="Pclass", y="Age", hue="Gender", data=df)
plt.title("Age Distribution Across Passenger Classes and Gender")
plt.show()

plt.figure(figsize=(10,6))

# Select only numerical columns
numerical_df = df.select_dtypes(include=["number"])

# Compute correlation and plot heatmap
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df, hue="Survived", diag_kind="kde")
plt.show()
```
<img width="489" height="401" alt="output 1" src="https://github.com/user-attachments/assets/272f36cc-4839-4771-8710-8c9d89dcbb68" />
<img width="794" height="345" alt="output 2" src="https://github.com/user-attachments/assets/b9ad093f-d833-4b15-8956-f8dc950a033b" />
<img width="367" height="515" alt="output 3" src="https://github.com/user-attachments/assets/ebf8a1d8-6b26-4502-85ae-c0a00613c576" />
<img width="571" height="455" alt="output 4" src="https://github.com/user-attachments/assets/4f72136d-78a7-4f9d-9522-b211a4971331" />
<img width="1386" height="506" alt="output 6" src="https://github.com/user-attachments/assets/afa13ce5-7d72-4f37-ae37-7e4d48c5cc00" />
<img width="582" height="512" alt="output 7" src="https://github.com/user-attachments/assets/7b02efba-0154-43e7-a9e3-aa4566b260cb" />
<img width="562" height="455" alt="output 8" src="https://github.com/user-attachments/assets/2adddc61-8a0e-44d8-941a-eab49bec0074" />
<img width="562" height="455" alt="output 9" src="https://github.com/user-attachments/assets/b5a1e98b-1f28-477f-a29f-4e0ccfeca053" />
<img width="768" height="528" alt="output 10" src="https://github.com/user-attachments/assets/abb0c091-ed07-45ba-b6ad-642f2e1f2238" />
<img width="1315" height="1231" alt="output 11" src="https://github.com/user-attachments/assets/536468e6-3815-4f6d-9375-ae4bad921972" />


## RESULT:
Thus, the Exploratory Data Analysis on the given data set was performed successfully.
