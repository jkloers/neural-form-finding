from src.utils.training_viz import plot_training_loss
history = [
    {'total': 100.0, 'chamfer': 50.0, 'energy': 50.0},
    {'total': 10.0, 'chamfer': 5.0, 'energy': 5.0}
]
plot_training_loss(history, show=False)
print("Plot generated successfully!")
