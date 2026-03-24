import pandas as pd 
import matplotlib.pyplot as plt 
  
df = pd.read_csv("Labels.csv") 

label_counts = df["Label"].value_counts()
ax = label_counts.plot(kind="bar", color="skyblue")
label_counts.plot(kind="bar")

plt.title("Glaucoma Label Distribution") 
plt.xlabel("Label") 
plt.ylabel("Number of Images")

for i, count in enumerate(label_counts):
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
plt.show()