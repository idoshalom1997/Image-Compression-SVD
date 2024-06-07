"""How few numbers does it take to keep a photo? Image compression with SVD.

Code exported from image_compression_svd.ipynb; see the notebook for outputs and charts.
"""

# %%
from PIL import Image

# Open the image file
image_path = 'data/Cliff_beach_BOEM_gov.jpg'  
image = Image.open(image_path)


# Get the dimensions of the image
width, height = image.size

# Display the dimensions
print(f'Dimensions of the image: Width = {width} pixels, Height = {height} pixels')

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert grayscale image to numpy array
gray_array = np.array(gray_image)

# Standardize the data
scaler = StandardScaler()
normalized_gray_array = scaler.fit_transform(gray_array)

# Convert normalized array back to image
normalized_gray_image = Image.fromarray(((normalized_gray_array - normalized_gray_array.min()) / (normalized_gray_array.max() - normalized_gray_array.min()) * 255).astype(np.uint8))

# Display the original grayscale image and the normalized grayscale image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original grayscale image
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

# Normalized grayscale image
axes[1].imshow(normalized_gray_image, cmap='gray')
axes[1].set_title('Normalized Grayscale Image')
axes[1].axis('off')

plt.show()

# %%
# Perform SVD on the normalized image
U, S, Vt = np.linalg.svd(normalized_gray_array, full_matrices=False)

# Number of singular values
num_singular_values = len(S)
print(f'Number of singular values: {num_singular_values}')

# %%
# Calculate the singular values percentage
singular_values_percentage = (S / S[0]) * 100

# Plot the singular values
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(S)) + 1, S, color='deeppink', alpha=0.5, width=2.5)

# Highlight singular values bigger than 1% of the first singular value 
threshold = 1
green_indices = np.where(singular_values_percentage > threshold)[0]
plt.bar(green_indices + 1, S[green_indices], color='lightgreen', alpha=0.7, width=2.5)

# Add a vertical dashed line to separate the singular values
plt.axvline(x=len(green_indices) + 1, color='black', linestyle='--')

# Add text on the dashed line
plt.text(len(green_indices) + 60, max(S), '<1%', ha='center', va='bottom', color='black', fontsize=12)

plt.xlabel('Index')
plt.ylabel('Singular Values')
plt.title('Singular Values In Decreasing Order')
plt.show()

print(f"There are {len(green_indices)} singular values bigger than 1% of the first singular value (including the first)")

# %%
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# Calculate the original size of the normalized image in bytes
original_size_bytes = normalized_gray_array.nbytes
print(f'Original normalized image size: {original_size_bytes} bytes')

# Number of PCs to use
pcs = [12, 24, 48, 96, len(green_indices)]

# Function to restore the image using a given number of principal components
def restore_image(U, S, Vt, num_pcs):
    U_reduced = U[:, :num_pcs]
    S_reduced = S[:num_pcs]
    Vt_reduced = Vt[:num_pcs, :]
    
    # Reconstruct the image
    restored_image = np.dot(U_reduced, np.dot(np.diag(S_reduced), Vt_reduced))
    
    # Normalize the restored image to fit in the 0-255 range
    restored_image_normalized = (restored_image - restored_image.min()) / (restored_image.max() - restored_image.min()) * 255
    
    return restored_image_normalized.astype(np.uint8)

# Function to calculate the compressed size of the image 
def calc_compressed_size(image_array):
    with io.BytesIO() as output:
        Image.fromarray(image_array).save(output, format='PNG')
        return output.tell()

# Plot the restored images
fig, axes = plt.subplots(1, len(pcs), figsize=(20, 5))
fig.suptitle('Restored Images with Different Numbers of PCs', size=16)

for i, num_pcs in enumerate(pcs):
    restored_image = restore_image(U, S, Vt, num_pcs)
    restored_size_bytes = calc_compressed_size(restored_image)
    axes[i].imshow(restored_image, cmap='gray')
    axes[i].set_title(f'{num_pcs} PCs\n Size: {restored_size_bytes} bytes')
    axes[i].axis('off')

plt.show()
