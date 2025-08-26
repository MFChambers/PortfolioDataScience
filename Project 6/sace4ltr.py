here is project 6 : --- title: "Coding Challenge - DS 250" subtitle: "Timed Quiz: Q1–Q4" author: "Maia Faith Chambers" format: html: self-contained: true toc: true toc-depth: 2 title-block-banner: true code-fold: true code-summary: "Show Code" code-tools: toggle: true caption: See Code execute: kernel: python312 warning: false ---
{python}
import sys
print(sys.executable)

url = "https://github.com/byuidatascience/data4names/raw/master/data-raw/names_year/names_year.csv"
names = pd.read_csv(url)

print(names.columns)
names.head()
{python}
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Might be needed for numerical operations

# 1. Load the data (replace with your actual data loading)
# Assuming 'names_data.csv' has columns 'Year', 'Line1_Data', 'Line2_Data', etc.
df = pd.read_csv('names_data.csv')

# 2. Create the plot
fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

# 3. Plot the lines
ax.plot(df['Year'], df['Line1_Data'], color='blue', label='Line 1')
ax.plot(df['Year'], df['Line2_Data'], color='green', label='Line 2')
ax.plot(df['Year'], df['Line3_Data'], color='red', label='Line 3')

# 4. Set titles and labels
ax.set_title('Chart Title Here', fontsize=16)
ax.set_xlabel('X-axis Label', fontsize=12)
ax.set_ylabel('Y-axis Label', fontsize=12)

# 5. Configure axes (example)
ax.set_xlim(min(df['Year']), max(df['Year'])) # Adjust limits if needed
ax.set_ylim(min(df['Line1_Data'].min(), df['Line2_Data'].min(), df['Line3_Data'].min()) * 0.9,
            max(df['Line1_Data'].max(), df['Line2_Data'].max(), df['Line3_Data'].max()) * 1.1)

# 6. Add annotations (example - you'll need to determine specific points and text)
ax.annotate('Annotation Text 1', xy=(df['Year'].iloc[5], df['Line1_Data'].iloc[5]), # Point to annotate
            xytext=(df['Year'].iloc[5] + 1, df['Line1_Data'].iloc[5] + 50), # Position for the text
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10, color='darkgray')

# Add more annotations for other lines as needed...

# 7. Display the plot
plt.legend() # Show the legend for the lines
plt.grid(True) # Add a grid for better readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
{python}
# Plot
plt.figure(figsize=(10,6))
sns.lineplot(data=names, x="year", y="n", hue="name")
plt.title("Name Popularity Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Babies")
plt.axvline(2000, color='gray', linestyle='--')
plt.text(2001, names['n'].max() * 0.8, "Millennium", rotation=90)
plt.tight_layout()
plt.show()
{python}
problem = pd.Series([np.nan, 18, 22, 45, 31, np.nan, 85, 38, 129, 8000, 22, 2])
std_val = problem.std()
filled = problem.fillna(std_val)
result = round(filled.mean(), 2)
print("Final Mean:", result)
{python}
ages = pd.Series(["10-25", "10-25", "26-35", "56-85", "0-9", "46-55",
                  "56-85", "0-9", "26-35", "56-85", "0-9", "10-25"])
age_df = ages.value_counts().reset_index()
age_df.columns = ["Age Range", "Count"]

plt.figure(figsize=(8,5))
sns.barplot(data=age_df, x="Age Range", y="Count", order=sorted(age_df["Age Range"]))
plt.title("Age Range Frequency")
plt.xlabel("Age Range")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
{python}
# Load cleaned Star Wars dataset
df = pd.read_csv("star_wars_clean.csv")
df = df.dropna()
y = df["female"]
X = pd.get_dummies(df.drop(columns=["female"]))

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2022)
model = RandomForestClassifier(random_state=2022)
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.2%}")

# Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10).reset_index()
top10.columns = ["Feature", "Importance"]

plt.figure(figsize=(8,5))
sns.barplot(data=top10, x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()
{python}
# Load cleaned Star Wars dataset
df = pd.read_csv("star_wars_clean.csv")
df = df.dropna()
y = df["female"]
X = pd.get_dummies(df.drop(columns=["female"]))

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2022)
model = RandomForestClassifier(random_state=2022)
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.2%}")

# Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10).reset_index()
top10.columns = ["Feature", "Importance"]

plt.figure(figsize=(8,5))
sns.barplot(data=top10, x="Importance", y="Feature", palette="viridis")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()