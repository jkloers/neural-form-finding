import os
import matplotlib.pyplot as plt

def plot_training_loss(history_loss, save_dir="data/outputs/runs/plots", show=True):
    """
    Plot the different components of the training loss.
    Uses the Princeton Form Finding Lab theme: Orange and Electric Green.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(len(history_loss))
    total_loss = [h['total'] for h in history_loss]
    chamfer_loss = [h['chamfer'] for h in history_loss]
    energy_loss = [h['energy'] for h in history_loss]
    
    # Aesthetic styling for scientific papers (White background)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors: Princeton Orange, Green, and Dark Gray for Total
    color_total = '#333333'     # Dark gray
    color_geom = '#F58025'      # Princeton Orange
    color_phys = '#009900'      # Readable Green
    
    ax.plot(epochs, total_loss, linestyle='-', linewidth=3, color=color_total, 
            label='Total Loss', alpha=0.9)
    ax.plot(epochs, chamfer_loss, linestyle='--', linewidth=2.5, color=color_geom, 
            label='Geometric (Chamfer)', alpha=0.9)
    ax.plot(epochs, energy_loss, linestyle=':', linewidth=2.5, color=color_phys, 
            label='Physical (Energy Penalty)', alpha=0.9)
    
    # Modern grid and labels
    ax.set_title("End-to-End Optimization Loss", fontsize=16, fontweight='bold', color='black', pad=20)
    ax.set_xlabel("Epoch", fontsize=12, color='black')
    ax.set_ylabel("Loss (Log Scale)", fontsize=12, color='black')
    ax.set_yscale('log')
    
    ax.grid(True, linestyle='--', color='#DDDDDD', alpha=0.7)
    
    # Legend aesthetics
    legend = ax.legend(fontsize=11, loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC')
    for text in legend.get_texts():
        text.set_color("black")
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(colors='black')
    
    save_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved detailed aesthetic loss curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
