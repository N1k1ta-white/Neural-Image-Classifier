import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Load the data
df = pd.read_csv('data.csv')

# Create a directory for saving plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Plot training and test accuracy over epochs
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['train_acc'], 'b-', label='Training Accuracy')
plt.plot(df['epoch'], df['test_acc'], 'r-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy vs Epochs')
plt.legend()
plt.grid(True)

# Add vertical line to mark the transition from with to without data augmentation
without_aug_start = df[df['phase'] == 'without data augmentation']['epoch'].min()
plt.axvline(x=without_aug_start-0.5, color='green', linestyle='--', 
           label=f'Switched to No Augmentation (Epoch {without_aug_start})')
plt.legend()
plt.savefig('plots/accuracy_vs_epochs.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Plot training and test loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
plt.plot(df['epoch'], df['test_loss'], 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs Epochs')
plt.axvline(x=without_aug_start-0.5, color='green', linestyle='--', 
           label=f'Switched to No Augmentation (Epoch {without_aug_start})')
plt.legend()
plt.grid(True)
plt.savefig('plots/loss_vs_epochs.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Create a scatter plot showing the relationship between training and test accuracy
plt.figure(figsize=(10, 8))
sns.scatterplot(x='train_acc', y='test_acc', hue='phase', data=df, s=100, alpha=0.7)
plt.xlabel('Training Accuracy (%)')
plt.ylabel('Test Accuracy (%)')
plt.title('Training vs Test Accuracy')
# Add the identity line
min_val = min(df['train_acc'].min(), df['test_acc'].min())
max_val = max(df['train_acc'].max(), df['test_acc'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Identity Line')
plt.legend()
plt.savefig('plots/train_vs_test_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Plot learning rate changes over time
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['lr'], 'g-o')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')  # Using log scale for better visualization
plt.grid(True)
plt.savefig('plots/learning_rate.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Plot execution time per epoch
plt.figure(figsize=(12, 6))
plt.bar(df['epoch'], df['time'], color=np.where(df['phase'] == 'with data augmentation', 'blue', 'orange'))
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Execution Time per Epoch')

# Add a legend for the colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', label='With Data Augmentation'),
    Patch(facecolor='orange', label='Without Data Augmentation')
]
plt.legend(handles=legend_elements)
plt.savefig('plots/execution_time.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Create heatmap of correlations between metrics
correlation_metrics = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'time']
corr_matrix = df[correlation_metrics].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Training Metrics')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Calculate and plot overfitting measure (difference between train and test accuracy)
df['overfitting_gap'] = df['train_acc'] - df['test_acc']
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['overfitting_gap'], 'purple')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Train Acc - Test Acc (%)')
plt.title('Potential Overfitting Measure (Train Accuracy - Test Accuracy)')
plt.axvline(x=without_aug_start-0.5, color='green', linestyle='--', 
           label=f'Switched to No Augmentation (Epoch {without_aug_start})')
plt.legend()
plt.grid(True)
plt.savefig('plots/overfitting_measure.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Plot moving average of test accuracy to show trend more clearly
window_size = 3
df['test_acc_ma'] = df['test_acc'].rolling(window=window_size).mean()
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['test_acc'], 'b-', alpha=0.4, label='Test Accuracy')
plt.plot(df['epoch'][window_size-1:], df['test_acc_ma'][window_size-1:], 'r-', 
         label=f'Moving Average (window={window_size})')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy Trend with Moving Average')
plt.legend()
plt.grid(True)
plt.savefig('plots/test_accuracy_trend.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Create a subplot dashboard with key metrics
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Training Dashboard', fontsize=16)

# Accuracy plot
axs[0, 0].plot(df['epoch'], df['train_acc'], 'b-', label='Train')
axs[0, 0].plot(df['epoch'], df['test_acc'], 'r-', label='Test')
axs[0, 0].set_title('Accuracy')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy (%)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Loss plot
axs[0, 1].plot(df['epoch'], df['train_loss'], 'b-', label='Train')
axs[0, 1].plot(df['epoch'], df['test_loss'], 'r-', label='Test')
axs[0, 1].set_title('Loss')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Learning rate
axs[1, 0].plot(df['epoch'], df['lr'], 'g-')
axs[1, 0].set_title('Learning Rate')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Learning Rate')
axs[1, 0].set_yscale('log')
axs[1, 0].grid(True)

# Time per epoch
axs[1, 1].bar(df['epoch'], df['time'], 
             color=np.where(df['phase'] == 'with data augmentation', 'blue', 'orange'))
axs[1, 1].set_title('Time per Epoch')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Time (seconds)')
handles = [
    Patch(facecolor='blue', label='With Aug'),
    Patch(facecolor='orange', label='Without Aug')
]
axs[1, 1].legend(handles=handles)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plots/dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# Display summary statistics
print("=== Training Data Summary ===")
print(f"Total epochs: {df['epoch'].max()}")
print(f"With data augmentation: {len(df[df['phase'] == 'with data augmentation'])} epochs")
print(f"Without data augmentation: {len(df[df['phase'] == 'without data augmentation'])} epochs")
print(f"Best test accuracy: {df['test_acc'].max():.2f}% (Epoch {df.loc[df['test_acc'].idxmax(), 'epoch']})")
print(f"Final test accuracy: {df.iloc[-1]['test_acc']:.2f}%")
print(f"Average epoch time with augmentation: {df[df['phase'] == 'with data augmentation']['time'].mean():.2f} seconds")
print(f"Average epoch time without augmentation: {df[df['phase'] == 'without data augmentation']['time'].mean():.2f} seconds")

# If running in a Jupyter notebook or similar environment, you can also show the plots
# plt.show()

print("\nAll plots have been saved to the 'plots' directory.")