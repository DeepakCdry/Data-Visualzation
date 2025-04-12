import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



def clean_ev_data(df):
    df.columns = df.columns.str.strip()               
    df = df.drop_duplicates()                         
    for col in ['City', 'State', 'Make', 'Model', 'Electric Vehicle Type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
    df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')
    df['Base MSRP'] = pd.to_numeric(df['Base MSRP'], errors='coerce')
    return df





path = "C:/Users/user/OneDrive/Desktop/4th Semester/INT375/Electric_Vehicle_Population_Data 1230.csv"
df = pd.read_csv(path)


df = clean_ev_data(df)

print("Columns in dataset:", df.columns.tolist())




print("\n--- First 5 Rows ---")
print(df.head(5))

print("\n--- Last 5 Rows ---")
print(df.tail(5))



# --- Plot 1: Most Common EV Makes
top_makes = df['Make'].value_counts().head(10)
plt.figure(figsize=(10, 6))


top_makes_df = pd.DataFrame({
    'Make': top_makes.index,
    'Count': top_makes.values
})

sns.barplot(data=top_makes_df, x='Count', y='Make', hue='Make', palette='coolwarm', legend=False)



plt.title('Top 10 EV Makes')
plt.xlabel('Count')
plt.ylabel('Make')
plt.tight_layout()
plt.show()




# --- Plot 2: Electric Range Distribution
if 'Electric Range' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Electric Range'].dropna(), bins=30, kde=True, color='teal')
    plt.title('Distribution of Electric Vehicle Ranges')
    plt.xlabel('Electric Range (miles)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


# --- Plot 3: EVs Over Model Year
plt.figure(figsize=(12, 6))


sns.countplot(data=df, x='Model Year', hue='Model Year', 
              order=sorted(df['Model Year'].dropna().unique()), palette='magma', legend=False)



plt.title('EV Count by Model Year')
plt.xticks(rotation=45)
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.tight_layout()
plt.show()









# 4. Select only 'Electric Range' and 'Base MSRP'
numeric_cols = ['Electric Range', 'Base MSRP']
df_numeric = df[numeric_cols].dropna()

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Plot heatmap using Seaborn (works perfectly inside Spyder)
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation between Electric Range and Base MSRP')
plt.tight_layout()
plt.show()
plt.show()




# 5. Filter data with non-null 'Model Year' and 'Base MSRP'
df_clean = df[['Model Year', 'Base MSRP']].dropna()

# Convert Model Year to integer if needed
df_clean['Model Year'] = df_clean['Model Year'].astype(int)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='Model Year', y='Base MSRP', color='darkorange', alpha=0.6, edgecolor='w')

plt.title('Scatter Plot: Model Year vs Base MSRP')
plt.xlabel('Model Year')
plt.ylabel('Base MSRP ($)')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.show()




# --- Plot 6: Box Plot - Electric Range by EV Type ---
filtered_df = df[df['Electric Vehicle Type'].isin(['Battery Electric Vehicle (Bev)', 'Plug-In Hybrid Electric Vehicle (Phev)'])]
filtered_df = filtered_df[['Electric Vehicle Type', 'Electric Range']].dropna()


print("\nFiltered Data for Box Plot:")
print(filtered_df.head())
print("Unique EV Types:", filtered_df['Electric Vehicle Type'].unique())


plt.figure(figsize=(8, 6))
sns.boxplot(data=filtered_df, x='Electric Vehicle Type', y='Electric Range', hue='Electric Vehicle Type', palette='Set2')
plt.title('Electric Range by Electric Vehicle Type')
plt.xlabel('Electric Vehicle Type')
plt.ylabel('Electric Range (miles)')
plt.tight_layout()
plt.show()






# 7.Clean and standardize 'Electric Vehicle Type' column
df['Electric Vehicle Type'] = df['Electric Vehicle Type'].str.title().str.strip()
df['Electric Vehicle Type'] = df['Electric Vehicle Type'].replace({
    'Battery Electric Vehicle (Bev)': 'BEV',
    'Plug-In Hybrid Electric Vehicle (Phev)': 'PHEV'
})

# Count EV types
ev_type_counts = df['Electric Vehicle Type'].value_counts()
print("EV Type Counts:\n", ev_type_counts)

# Plot pie chart
plt.figure(figsize=(6, 6))
colors = ['mediumseagreen', 'skyblue']
plt.pie(ev_type_counts, labels=ev_type_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)

plt.title('Distribution of Electric Vehicle Types')
plt.axis('equal')  # Makes the pie chart circular
plt.tight_layout()
plt.show()




# 8. Count number of EVs per Model Year
ev_trend = df['Model Year'].value_counts().sort_index()

# Plotting the line chart
plt.figure(figsize=(10, 6))
plt.plot(ev_trend.index, ev_trend.values, marker='o', linestyle='-', color='royalblue')

plt.title('Electric Vehicles Registered Over the Years')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.grid(True)
plt.tight_layout()
plt.show()



# ----9--Step 1: Filter relevant numeric columns and drop nulls
df_reg = df[['Model Year', 'Electric Range', 'Base MSRP']].dropna()

# Step 2: Define features (X) and target (y)
X = df_reg[['Model Year', 'Electric Range']]
y = df_reg['Base MSRP']

# Step 3: Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))
