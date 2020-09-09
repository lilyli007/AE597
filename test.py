import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
df = pd.DataFrame(x,y)

sns.set()
sns.lineplot(x=x, y=y)
plt.show()