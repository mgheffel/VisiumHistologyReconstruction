import torch
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

def visualize_gene_expression( img_tensor, pixel_coords,  adata, gene_idx, gene_name=None):

    # Determine gene name if not provided
    if gene_name is None:
        gene_name = adata.var.index[gene_idx] if 'index' in adata.var else f"Gene {gene_idx}"

    # Prepare the original image for plotting (convert back to H, W, C)
    original_image = img_tensor.permute(1, 2, 0).numpy()  # Shape: (H, W, 3)
    if original_image.max() <= 1.0:
        original_image = original_image * 255.0  # Scale back to [0, 255] if normalized
    original_image = original_image.astype(np.uint8)

    # Predicted gene expression map for the selected gene
    pred_map = adata.to_df()[gene_name] # Shape: (H, W)
    # Create the visualization
    plt.figure(figsize=(4, 4))

    # Plot the original histology image
    plt.imshow(original_image)

    # Overlay the predicted gene expression map with transparency
#     plt.imshow(pred_map, cmap='inferno', alpha=0.5)  # Alpha controls transparency

    # Overlay ground truth at spot locations with smaller dots
    plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1] * -1, c=pred_map, cmap='viridis', 
                s=10, edgecolors='w', linewidth=0.5, label='Ground Truth')  # Reduced from s=50 to s=10

    # Add a colorbar for the predicted map
    plt.colorbar(label=f'Predicted Expression ({gene_name})')

    # Add title and labels
    plt.title(f'Gene Expression Prediction for {gene_name}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.figure(figsize=(4, 4))
    # Show the plot
    plt.show()
    sc.pl.umap(adata,color=gene_name)
    
def visualize_gene_expression_4genes(img_tensor, pixel_coords, adata, gene_indices, gene_names=None):
    """
    Visualize gene expression for multiple genes side by side.
    
    Parameters:
    - img_tensor: Tensor of shape (C, H, W) containing the original image
    - pixel_coords: Array of shape (n_spots, 2) with x, y coordinates
    - adata: AnnData object containing gene expression data
    - gene_indices: List of gene indices to visualize
    - gene_names: Optional list of gene names (defaults to adata.var.index or "Gene {idx}")
    """
    # Ensure we have 4 genes to plot
    if len(gene_indices) != 4:
        raise ValueError("Please provide exactly 4 gene indices")

    # Determine gene names if not provided
    if gene_names is None:
        gene_names = [adata.var.index[i] if 'index' in adata.var else f"Gene {i}" 
                     for i in gene_indices]
    elif len(gene_names) != 4:
        raise ValueError("Please provide exactly 4 gene names")

    # Prepare the original image (convert back to H, W, C)
    original_image = img_tensor.permute(1, 2, 0).numpy()  # Shape: (H, W, 3)
    if original_image.max() <= 1.0:
        original_image = original_image * 255.0  # Scale back to [0, 255] if normalized
    original_image = original_image.astype(np.uint8)

    # Create figure with 4 subplots side by side
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))

    # Plot each gene's expression
    for i, (gene_idx, gene_name) in enumerate(zip(gene_indices, gene_names)):
        # Predicted gene expression map for the selected gene
        pred_map = adata.to_df()[gene_name]  # Shape: (n_spots,)

        # Plot the original histology image
        axs[i].imshow(original_image)

        # Overlay ground truth at spot locations with smaller dots
        scatter = axs[i].scatter(pixel_coords[:, 0], pixel_coords[:, 1] * -1, 
                                c=pred_map, cmap='viridis', 
                                s=10, edgecolors='w', linewidth=0.5, 
                                label='Ground Truth')

        # Add a colorbar for this subplot
        plt.colorbar(scatter, ax=axs[i], label=f'Expression ({gene_name})')

        # Add title and remove axes
        axs[i].set_title(f'{gene_name}', pad=10)
        axs[i].axis('off')

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    # Optional: Show UMAP plots for all genes
    sc.pl.umap(adata, color=gene_names)