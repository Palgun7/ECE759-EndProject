import numpy as np
import matplotlib.pyplot as plt

def pca(X, num_components):
    """
    Perform Principal Component Analysis (PCA) and reconstruct the data.
    
    Parameters:
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    num_components : int
        Number of principal components to retain.

    Returns:
    X_reduced : np.ndarray
        Transformed data in the principal component space.
    X_reconstructed : np.ndarray
        Reconstructed data after PCA.
    """
    # Step 1: Center the data
    X_mean = np.mean(X, axis=0)
    X_meaned = X - X_mean

    # Step 2: Compute the covariance matrix
    covariance_matrix = np.cov(X_meaned, rowvar=False)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top 'num_components' eigenvectors
    principal_components = eigenvectors[:, :num_components]

    # Step 6: Project data onto principal components
    X_reduced = np.dot(X_meaned, principal_components)

    # Step 7: Reconstruct the data
    X_reconstructed = np.dot(X_reduced, principal_components.T) + X_mean

    return X_reduced, X_reconstructed

def plot_images(original, reconstructed, n_images=5):
    """
    Plot original and reconstructed images side by side.

    Parameters:
    original : np.ndarray
        Original dataset (each row is an image).
    reconstructed : np.ndarray
        Reconstructed dataset (each row is an image).
    n_images : int
        Number of images to display.
    """
    plt.figure(figsize=(10, 4))
    for i in range(n_images):
        # Original Image
        plt.subplot(2, n_images, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed Image
        plt.subplot(2, n_images, n_images + i + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load an example dataset (MNIST handwritten digits, e.g., using sklearn or custom loader)
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data / 255.0  # Normalize data to [0, 1]

    # Perform PCA
    num_components = 50  # Retain top 50 components
    X_reduced, X_reconstructed = pca(data.values, num_components)

    # Visualize original vs reconstructed images
    plot_images(data.values, X_reconstructed, n_images=5)
