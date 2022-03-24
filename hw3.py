from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    raw_data = np.load(filename)
    x = raw_data - np.mean(raw_data, axis = 0)
    return x

def get_covariance(dataset):
    result = np.dot(np.transpose(dataset), dataset)/2413
    return result

def get_eig(S, m):
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[1024-m,1023])
    sorting_indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorting_indices]
    eigenvectors = eigenvectors[:,sorting_indices]
    eigenvalues = np.diag(eigenvalues)
    return eigenvalues, eigenvectors

def get_eig_prop(S, prop):
    temp_eigen = eigh(S, eigvals_only=True) 
    sum = np.sum(temp_eigen)
    limit = sum*prop
    eigenvalues, eigenvectors = eigh(S, subset_by_value=(limit, np.inf)) 
    sorting_indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorting_indices]
    eigenvectors = eigenvectors[:,sorting_indices]
    eigenvalues = np.diag(eigenvalues)
    return eigenvalues, eigenvectors

def project_image(image, U):
    rows, m = np.shape(U)
    alpha = np.zeros(m)
    projection = np.zeros(rows)
    for i in range(m):
        alpha[i] = np.dot(np.transpose(U[:, i]), image)
        projection = projection + np.transpose(np.multiply(U[:,i],alpha[i]))
    return projection

def display_image(orig, proj):
    image = np.reshape(orig, (32,32))
    image = np.transpose(image)
    proj = np.reshape(proj, (32,32))
    proj = np.transpose(proj)
    fig, axs = plt.subplots(figsize = (7,2), ncols=2)
    axs[0].set_title("Original")
    axs[1].set_title("Projection")
    image1 = axs[0].imshow(image, aspect = 'equal')
    image2 = axs[1].imshow(proj, aspect = 'equal')
    fig.colorbar(image1, ax=axs[0])
    fig.colorbar(image2, ax=axs[1])
    plt.show()
