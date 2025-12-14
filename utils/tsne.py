import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

class_names = {
    0: "Ring",
    1: "Trophozoite",
    2: "Schizont",
    3: "Gametocyte",
    4: "Healthy"
}

def tsne_visualize(X, y,
                   save_path: str):


    # 2. Configure the t-SNE model
    # n_components=2 for a 2D visualization
    # perplexity usually between 5 and 50
    # random_state for reproducibility
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

    # 3. Fit and transform the data
    # t-SNE only has a fit_transform method, not separate fit/transform methods
    X_reduced = tsne.fit_transform(X)

    # 4. Visualize the results
    plt.figure(figsize=(8, 6))

    for label in np.unique(y):
        idx = y == label
        plt.scatter(
            X_reduced[idx, 0],
            X_reduced[idx, 1],
            s=30,
            alpha=0.7,
            label=class_names.get(label, f"Class {label}")
        )

    # plt.title('t-SNE visualization of model features')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.legend(markerscale=1, fontsize=10)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()
