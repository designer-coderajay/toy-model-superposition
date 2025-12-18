"""
Toy Model of Superposition
==========================
Replicating the core experiment from Anthropic's "Toy Models of Superposition" paper.

THE CORE INSIGHT:
----------------
Neural networks can represent N features in M dimensions (where N > M) by exploiting
sparsity. If features rarely co-occur, they can be stored as non-orthogonal vectors
that "interfere" but rarely cause problems in practice.

In 2D with 5 sparse features, the optimal solution forms a PENTAGON.
Each feature vector points to a vertex, maximizing the minimum angle between any two.

WHAT THIS CODE DOES:
-------------------
1. Generate synthetic data with sparse features (each feature active with probability p)
2. Train a linear autoencoder: input (5D) -> bottleneck (2D) -> output (5D)
3. The encoder weights W reveal HOW the model represents 5 features in 2D
4. Visualize W to see the geometric structure emerge

Author: Ajay
Project: ML Portfolio - Mechanistic Interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

# ============================================================================
# PART 1: UNDERSTANDING THE MATH
# ============================================================================
"""
THE SETUP:
- We have n_features = 5 "ground truth" features
- Each feature has an "importance" weight (how much we care about reconstructing it)
- Features are sparse: each is active with probability p (sparsity)

THE MODEL:
- Linear autoencoder: x -> Wx -> W^T(Wx) -> x_reconstructed
- W is shape (n_hidden, n_features) = (2, 5)
- Each COLUMN of W is how that feature is represented in the 2D space

THE LOSS:
- Weighted MSE: sum over features of importance[i] * (x[i] - x_hat[i])^2
- This means the model cares more about reconstructing important features

THE KEY INSIGHT:
- If features were dense (always active), W would learn to represent only
  the top 2 most important features perfectly (orthogonal basis)
- But if features are SPARSE, W can "superpose" all 5 features as nearly-orthogonal
  vectors, accepting small reconstruction errors when multiple features co-occur
"""


class ToyModelSuperposition:
    """
    A simple linear autoencoder that demonstrates superposition.
    
    Architecture: x (n_features) -> W -> hidden (n_hidden) -> W^T -> x_hat (n_features)
    
    The weight matrix W shows how features are embedded in the lower-dimensional space.
    With sparse features, W learns geometric arrangements (pentagons, etc.) to pack
    more features than dimensions.
    """
    
    def __init__(
        self,
        n_features: int = 5,
        n_hidden: int = 2,
        importance_decay: float = 0.7,
        seed: Optional[int] = 42
    ):
        """
        Args:
            n_features: Number of input features (default 5)
            n_hidden: Bottleneck dimension (default 2, so we can visualize)
            importance_decay: How much less important each subsequent feature is.
                              importance[i] = importance_decay^i
                              With 0.7: [1.0, 0.7, 0.49, 0.34, 0.24]
            seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        
        if seed is not None:
            np.random.seed(seed)
        
        # Feature importance: exponentially decaying
        # This creates a "priority" - feature 0 is most important
        self.importance = np.array([importance_decay ** i for i in range(n_features)])
        
        # Initialize weights: W is (n_hidden, n_features)
        # Each COLUMN of W is the 2D representation of that feature
        # Xavier initialization for stability
        scale = np.sqrt(2.0 / (n_features + n_hidden))
        self.W = np.random.randn(n_hidden, n_features) * scale
        
        # For tracking training
        self.loss_history = []
        
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Project input to hidden dimension: h = Wx"""
        return self.W @ x.T  # (n_hidden, batch_size)
    
    def decode(self, h: np.ndarray) -> np.ndarray:
        """Reconstruct from hidden: x_hat = W^T h"""
        return (self.W.T @ h).T  # (batch_size, n_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass: x -> h -> x_hat"""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat
    
    def compute_loss(self, x: np.ndarray, x_hat: np.ndarray) -> float:
        """
        Weighted MSE loss.
        
        Loss = mean over batch of: sum over features of importance[i] * (x[i] - x_hat[i])^2
        
        The importance weighting means the model prioritizes reconstructing
        high-importance features, but will still try to reconstruct low-importance
        ones if it can do so "for free" via superposition.
        """
        # (batch_size, n_features)
        squared_error = (x - x_hat) ** 2
        # Weight by importance
        weighted_error = squared_error * self.importance  # broadcasts over batch
        # Mean over batch and sum over features
        return np.mean(np.sum(weighted_error, axis=1))
    
    def compute_gradients(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. W analytically.
        
        For linear autoencoder with tied weights:
        x_hat = W^T W x
        Loss = sum_i importance[i] * (x_i - x_hat_i)^2
        
        dL/dW = 2 * (W @ x.T @ x @ diag(importance) - W @ x.T @ x @ W^T @ W @ diag(importance)
                     + W @ diag(importance) @ x.T @ x @ W^T @ W - diag(importance) @ x.T @ x @ W^T)
        
        Actually, let's derive this more carefully...
        
        Let h = Wx (hidden activations)
        Let x_hat = W^T h = W^T W x
        
        Loss L = (1/N) sum_batch sum_features importance[f] * (x[f] - x_hat[f])^2
        
        dL/dx_hat = -2 * importance * (x - x_hat) / N
        
        For W, we have two paths: W appears in both encode and decode
        
        dx_hat/dW (through decode) = d(W^T h)/dW, treating h as constant
        dx_hat/dW (through encode) = d(W^T W x)/dW
        
        This gets messy. Let's just use numerical gradients or compute it directly.
        
        Actually, the cleanest formulation:
        x_hat = W^T @ W @ x.T  (treating x as column vectors)
        
        Let's compute gradient numerically for correctness, then optimize.
        """
        # Compute gradient numerically (cleaner for this educational code)
        eps = 1e-5
        grad = np.zeros_like(self.W)
        
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                # Positive perturbation
                self.W[i, j] += eps
                x_hat_plus = self.forward(x)
                loss_plus = self.compute_loss(x, x_hat_plus)
                
                # Negative perturbation
                self.W[i, j] -= 2 * eps
                x_hat_minus = self.forward(x)
                loss_minus = self.compute_loss(x, x_hat_minus)
                
                # Restore
                self.W[i, j] += eps
                
                # Central difference
                grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad
    
    def compute_gradients_analytical(self, x: np.ndarray) -> np.ndarray:
        """
        Analytical gradient computation (faster than numerical).
        
        For L = sum_i w_i (x_i - [W^T W x]_i)^2
        
        Let's work through this:
        - h = W @ x.T  -> shape (n_hidden, batch_size)
        - x_hat = W.T @ h  -> shape (n_features, batch_size)
        - error = x.T - x_hat  -> shape (n_features, batch_size)
        - weighted_error = importance[:, None] * error  -> shape (n_features, batch_size)
        
        dL/dW has contributions from both W in encoder and W^T in decoder:
        
        1) From decoder (W^T @ h):
           dL/dW^T = -2 * weighted_error @ h.T / batch_size
           -> dL/dW contribution = (-2 * weighted_error @ h.T / batch_size).T
                                 = -2 * h @ weighted_error.T / batch_size
        
        2) From encoder (W @ x.T):
           dh/dW @ dL/dh
           dL/dh = W @ (-2 * weighted_error) / batch_size
           This contributes: dL/dh @ x = -2 * W @ weighted_error @ x / batch_size
        
        Let me verify with the chain rule more carefully...
        """
        batch_size = x.shape[0]
        
        # Forward pass
        h = self.W @ x.T  # (n_hidden, batch_size)
        x_hat_T = self.W.T @ h  # (n_features, batch_size)
        
        # Error and weighted error
        error = x.T - x_hat_T  # (n_features, batch_size)
        weighted_error = self.importance[:, None] * error  # (n_features, batch_size)
        
        # Gradient from decoder path: d/dW of W.T @ h, where h is treated as input
        # x_hat = W.T @ h
        # dL/dW.T = dL/dx_hat @ dxhat/dW.T = -2*weighted_error @ h.T
        # dL/dW (from this) = (dL/dW.T).T = h @ (-2*weighted_error).T = -2 * h @ weighted_error.T
        grad_from_decode = -2 * h @ weighted_error.T / batch_size  # (n_hidden, n_features)
        
        # Gradient from encoder path: d/dW of h = W @ x.T
        # dL/dh = dL/dx_hat @ dx_hat/dh = W @ (-2 * weighted_error)
        dL_dh = self.W @ (-2 * weighted_error / batch_size)  # (n_hidden, batch_size)
        # dh/dW: for each h[i] = sum_j W[i,j] * x[j], dh[i]/dW[i,j] = x[j]
        # So dL/dW[i,j] = sum_batch dL/dh[i] * x[j]
        grad_from_encode = dL_dh @ x  # (n_hidden, n_features)
        
        return grad_from_decode + grad_from_encode
    
    def train_step(self, x: np.ndarray, lr: float = 0.01) -> float:
        """Single training step with gradient descent."""
        x_hat = self.forward(x)
        loss = self.compute_loss(x, x_hat)
        
        # Use analytical gradients (much faster)
        grad = self.compute_gradients_analytical(x)
        
        # Gradient descent
        self.W -= lr * grad
        
        return loss
    
    def train(
        self,
        sparsity: float = 0.05,
        n_steps: int = 10000,
        batch_size: int = 256,
        lr: float = 0.01,
        print_every: int = 1000,
        seed: Optional[int] = None
    ) -> None:
        """
        Train the model on synthetic sparse data.
        
        Args:
            sparsity: Probability that each feature is active (0.05 = 5% active)
            n_steps: Number of training steps
            batch_size: Samples per step
            lr: Learning rate
            print_every: Print loss every N steps
            seed: Random seed for data generation
        """
        if seed is not None:
            np.random.seed(seed)
            
        print(f"Training with sparsity={sparsity} ({sparsity*100:.1f}% feature activation)")
        print(f"Feature importance: {self.importance.round(3)}")
        print("-" * 50)
        
        self.loss_history = []
        
        for step in range(n_steps):
            # Generate sparse batch
            x = self.generate_sparse_data(batch_size, sparsity)
            
            # Train step
            loss = self.train_step(x, lr)
            self.loss_history.append(loss)
            
            if (step + 1) % print_every == 0:
                print(f"Step {step + 1:5d} | Loss: {loss:.6f}")
        
        print("-" * 50)
        print(f"Final loss: {self.loss_history[-1]:.6f}")
    
    def generate_sparse_data(self, batch_size: int, sparsity: float) -> np.ndarray:
        """
        Generate synthetic sparse feature data.
        
        Each feature is independently:
        - Active (value sampled uniform [0, 1]) with probability 'sparsity'
        - Inactive (value = 0) with probability 1 - sparsity
        
        Args:
            batch_size: Number of samples
            sparsity: Probability of each feature being active
            
        Returns:
            x: shape (batch_size, n_features)
        """
        # Mask: which features are active
        mask = np.random.random((batch_size, self.n_features)) < sparsity
        
        # Values: uniform [0, 1] where active, 0 elsewhere
        values = np.random.random((batch_size, self.n_features))
        
        return mask * values
    
    def get_feature_vectors(self) -> np.ndarray:
        """
        Get the 2D representation of each feature.
        
        Returns columns of W, where each column is how that feature
        is represented in the hidden space.
        
        Returns:
            vectors: shape (n_features, n_hidden) - each row is a feature's 2D vector
        """
        return self.W.T  # Transpose so each ROW is a feature
    
    def analyze_geometry(self) -> dict:
        """
        Analyze the geometric structure of learned representations.
        
        For a pentagon (optimal for 5 features in 2D), we expect:
        - All vectors to have similar magnitudes
        - Angles between adjacent vectors ≈ 72° (360°/5)
        - Vectors evenly distributed around the origin
        """
        vectors = self.get_feature_vectors()
        
        # Compute magnitudes
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        # Compute all pairwise angles
        n = len(vectors)
        angles = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    cos_angle = np.dot(vectors[i], vectors[j]) / (magnitudes[i] * magnitudes[j] + 1e-8)
                    cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
                    angles[i, j] = np.arccos(cos_angle) * 180 / np.pi
        
        # Compute angle each vector makes with positive x-axis
        angles_from_x = np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / np.pi
        
        # Sort by angle to see if they're evenly distributed
        sorted_angles = np.sort(angles_from_x)
        angle_gaps = np.diff(sorted_angles)
        # Add wraparound gap
        angle_gaps = np.append(angle_gaps, 360 + sorted_angles[0] - sorted_angles[-1])
        
        return {
            'magnitudes': magnitudes,
            'pairwise_angles': angles,
            'angles_from_x_axis': angles_from_x,
            'sorted_angles': sorted_angles,
            'angle_gaps': angle_gaps,
            'mean_gap': np.mean(angle_gaps),
            'std_gap': np.std(angle_gaps),
            'ideal_gap': 360 / n  # For perfect polygon
        }


def visualize_superposition(
    model: ToyModelSuperposition,
    title: str = "Learned Feature Representations",
    show_analysis: bool = True
) -> plt.Figure:
    """
    Visualize the learned weight vectors showing superposition.
    
    Each feature is represented as a vector from origin.
    With 5 features in 2D and sufficient sparsity, expect a pentagon.
    """
    vectors = model.get_feature_vectors()
    importance = model.importance
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if show_analysis else 1, figsize=(14 if show_analysis else 7, 6))
    
    if show_analysis:
        ax_main, ax_loss = axes
    else:
        ax_main = axes
    
    # Main plot: feature vectors
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(vectors)))
    
    # Find plot limits
    max_val = np.max(np.abs(vectors)) * 1.3
    
    # Draw unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax_main.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, label='Unit circle')
    
    # Draw each feature vector
    for i, (vec, color) in enumerate(zip(vectors, colors)):
        # Arrow from origin
        ax_main.annotate(
            '', xy=vec, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=2)
        )
        
        # Label at arrow tip
        offset = vec / np.linalg.norm(vec) * 0.15 if np.linalg.norm(vec) > 0.1 else np.array([0.1, 0.1])
        ax_main.annotate(
            f'F{i}\n(imp={importance[i]:.2f})',
            xy=vec + offset,
            fontsize=9,
            ha='center',
            color=color,
            fontweight='bold'
        )
    
    # Draw polygon connecting features (sorted by angle)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_vectors = vectors[sorted_indices]
    
    # Close the polygon
    polygon = np.vstack([sorted_vectors, sorted_vectors[0]])
    ax_main.plot(polygon[:, 0], polygon[:, 1], 'b-', alpha=0.3, lw=1)
    ax_main.fill(polygon[:, 0], polygon[:, 1], alpha=0.1, color='blue')
    
    ax_main.set_xlim(-max_val, max_val)
    ax_main.set_ylim(-max_val, max_val)
    ax_main.set_aspect('equal')
    ax_main.axhline(y=0, color='k', linewidth=0.5)
    ax_main.axvline(x=0, color='k', linewidth=0.5)
    ax_main.set_xlabel('Hidden Dimension 1')
    ax_main.set_ylabel('Hidden Dimension 2')
    ax_main.set_title(title)
    ax_main.grid(True, alpha=0.3)
    
    # Analysis subplot: loss curve
    if show_analysis and model.loss_history:
        ax_loss.plot(model.loss_history)
        ax_loss.set_xlabel('Training Step')
        ax_loss.set_ylabel('Weighted MSE Loss')
        ax_loss.set_title('Training Loss')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')
    
    plt.tight_layout()
    return fig


def run_sparsity_comparison(sparsities: list = [0.01, 0.05, 0.1, 0.5, 1.0]) -> plt.Figure:
    """
    Show how sparsity affects the learned representations.
    
    Key insight:
    - High sparsity (p=0.01): Clear pentagon, model uses superposition
    - Low sparsity (p=0.5): Model can't superpose, focuses on top 2 features
    - Dense (p=1.0): Pure dimensionality reduction, only 2 features represented
    """
    fig, axes = plt.subplots(1, len(sparsities), figsize=(4*len(sparsities), 4))
    
    for ax, sparsity in zip(axes, sparsities):
        # Train model
        model = ToyModelSuperposition(n_features=5, n_hidden=2, seed=42)
        model.train(sparsity=sparsity, n_steps=5000, print_every=10000)
        
        vectors = model.get_feature_vectors()
        importance = model.importance
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(vectors)))
        
        # Plot
        max_val = max(1.5, np.max(np.abs(vectors)) * 1.3)
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2)
        
        # Feature vectors
        for i, (vec, color) in enumerate(zip(vectors, colors)):
            ax.annotate(
                '', xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2)
            )
            # Small label
            if np.linalg.norm(vec) > 0.1:
                offset = vec / np.linalg.norm(vec) * 0.12
                ax.annotate(f'{i}', xy=vec + offset, fontsize=8, ha='center', color=color)
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(f'Sparsity = {sparsity}\n({sparsity*100:.0f}% active)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Sparsity on Feature Representations', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def run_feature_sweep(n_features_list: list = [3, 4, 5, 6, 8]) -> plt.Figure:
    """
    Show different geometric shapes for different numbers of features.
    
    - 3 features in 2D -> Triangle (Mercedes logo shape)
    - 4 features in 2D -> Square
    - 5 features in 2D -> Pentagon
    - 6 features in 2D -> Hexagon
    - 8 features in 2D -> Octagon
    """
    fig, axes = plt.subplots(1, len(n_features_list), figsize=(4*len(n_features_list), 4))
    
    for ax, n_features in zip(axes, n_features_list):
        # Train model
        model = ToyModelSuperposition(n_features=n_features, n_hidden=2, seed=42)
        model.train(sparsity=0.03, n_steps=8000, print_every=10000)
        
        vectors = model.get_feature_vectors()
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(vectors)))
        
        # Plot
        max_val = max(1.5, np.max(np.abs(vectors)) * 1.3)
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2)
        
        # Feature vectors
        for i, (vec, color) in enumerate(zip(vectors, colors)):
            ax.annotate(
                '', xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2)
            )
        
        # Draw polygon
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_vectors = vectors[sorted_indices]
        polygon = np.vstack([sorted_vectors, sorted_vectors[0]])
        ax.plot(polygon[:, 0], polygon[:, 1], 'b-', alpha=0.3, lw=1)
        ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.1, color='blue')
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        shape_names = {3: 'Triangle', 4: 'Square', 5: 'Pentagon', 6: 'Hexagon', 8: 'Octagon'}
        shape_name = shape_names.get(n_features, f'{n_features}-gon')
        ax.set_title(f'{n_features} features\n({shape_name})')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Geometric Structures with Different Feature Counts', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def run_uniform_vs_decaying_importance() -> plt.Figure:
    """
    Compare uniform importance (perfect geometry) vs decaying importance (realistic).
    
    This is a KEY insight:
    - Uniform importance: all features equally valuable -> perfect polygon
    - Decaying importance: model allocates capacity to important features first
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    configs = [
        ("Uniform Importance\n(All features equal)", 1.0),  # decay=1.0 means all equal
        ("Decaying Importance\n(Realistic scenario)", 0.7)
    ]
    
    for ax, (title, decay) in zip(axes, configs):
        model = ToyModelSuperposition(n_features=5, n_hidden=2, importance_decay=decay, seed=42)
        model.train(sparsity=0.03, n_steps=10000, print_every=20000)
        
        vectors = model.get_feature_vectors()
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(vectors)))
        
        max_val = max(1.5, np.max(np.abs(vectors)) * 1.3)
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2)
        
        # Feature vectors
        for i, (vec, color) in enumerate(zip(vectors, colors)):
            ax.annotate(
                '', xy=vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5)
            )
            if np.linalg.norm(vec) > 0.1:
                offset = vec / np.linalg.norm(vec) * 0.15
                ax.annotate(f'F{i}', xy=vec + offset, fontsize=10, ha='center', 
                           color=color, fontweight='bold')
        
        # Draw polygon
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_vectors = vectors[sorted_indices]
        polygon = np.vstack([sorted_vectors, sorted_vectors[0]])
        ax.plot(polygon[:, 0], polygon[:, 1], 'b-', alpha=0.4, lw=1.5)
        ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.15, color='blue')
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Feature Importance on Geometric Structure', fontsize=14)
    plt.tight_layout()
    return fig


def create_interactive_demo() -> None:
    """
    Print explanation of what's happening for portfolio documentation.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TOY MODEL OF SUPERPOSITION                            ║
║                   Understanding Neural Network Representations                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT IS SUPERPOSITION?
─────────────────────
Neural networks often need to represent MORE features than they have dimensions.
For example: represent 5 features using only 2 neurons.

Mathematically, you can only have 2 orthogonal vectors in 2D space.
So how does the network "cheat"?

THE KEY INSIGHT:
───────────────
If features are SPARSE (rarely active together), the network can represent them
as NEARLY-orthogonal vectors. When features don't co-occur, interference doesn't
matter!

THE GEOMETRIC RESULT:
───────────────────
• 5 features in 2D → Pentagon (vertices at 72° apart)
• 6 features in 2D → Hexagon
• N features in 2D → N-gon

WHY THIS MATTERS:
───────────────
This is the foundation of mechanistic interpretability research:
1. Real neural networks use superposition extensively
2. This makes interpreting neurons difficult (they represent multiple features)
3. Sparse autoencoders try to "undo" superposition to find interpretable features

EXPERIMENTS IN THIS CODE:
───────────────────────
1. Pentagon formation with sparse features
2. Effect of sparsity (sparse → pentagon, dense → only 2 features)
3. Different geometries (triangle, square, pentagon, hexagon, octagon)
4. Uniform vs decaying feature importance
""")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TOY MODEL OF SUPERPOSITION")
    print("=" * 60)
    
    # Print educational overview
    create_interactive_demo()
    
    # -------------------------------------------------------------------------
    # Experiment 1: Basic Pentagon Formation
    # -------------------------------------------------------------------------
    print("\nEXPERIMENT 1: Learning a Pentagon (5 features in 2D)")
    print("=" * 60)
    
    model = ToyModelSuperposition(n_features=5, n_hidden=2, seed=42)
    model.train(sparsity=0.05, n_steps=10000, batch_size=256, lr=0.01, print_every=2000)
    
    # Analyze geometry
    analysis = model.analyze_geometry()
    print("\nGeometric Analysis:")
    print(f"  Vector magnitudes: {analysis['magnitudes'].round(3)}")
    print(f"  Angle gaps between adjacent features: {analysis['angle_gaps'].round(1)}°")
    print(f"  Mean gap: {analysis['mean_gap']:.1f}° (ideal for pentagon: {analysis['ideal_gap']:.1f}°)")
    print(f"  Gap std: {analysis['std_gap']:.1f}° (lower = more uniform)")
    
    # Save visualization
    fig1 = visualize_superposition(model, "Pentagon: 5 Features in 2D (Sparsity=5%)")
    fig1.savefig('pentagon_visualization.png', dpi=150, bbox_inches='tight')
    print("\nSaved: pentagon_visualization.png")
    
    # -------------------------------------------------------------------------
    # Experiment 2: Uniform vs Decaying Importance (KEY DEMO)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Uniform vs Decaying Feature Importance")
    print("=" * 60)
    
    fig_importance = run_uniform_vs_decaying_importance()
    fig_importance.savefig('importance_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: importance_comparison.png")
    
    # -------------------------------------------------------------------------
    # Experiment 3: Sparsity Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Effect of Sparsity")
    print("=" * 60)
    
    fig2 = run_sparsity_comparison([0.01, 0.05, 0.2, 0.5, 1.0])
    fig2.savefig('sparsity_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: sparsity_comparison.png")
    
    # -------------------------------------------------------------------------
    # Experiment 4: Different Feature Counts
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Different Geometric Shapes")
    print("=" * 60)
    
    fig3 = run_feature_sweep([3, 4, 5, 6, 8])
    fig3.savefig('geometric_shapes.png', dpi=150, bbox_inches='tight')
    print("Saved: geometric_shapes.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    
    plt.show()
