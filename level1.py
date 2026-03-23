import matplotlib
matplotlib.use('TkAgg') # Fixes the window display error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # 1. Load the data
    df = pd.read_csv('iris.csv', sep=r'\s+')
    
    # 2. Set the professional style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 3. Create the scatter plot
    sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', s=100, palette='bright')
    
    # 4. Add titles and labels
    plt.title('Codveda Project: Iris Species Data Visualization', fontsize=15)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    
    print("✅ Success! Level 1 visualization generated.")
    plt.show()

except Exception as e:
    print(f"❌ Error: {e}")