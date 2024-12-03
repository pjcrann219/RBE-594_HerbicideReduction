import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
import torch
from eval_utilities import load_model
from Utilities import get_dataloader, CustomDataset
import json
import os
from PIL import Image
import re
from matplotlib.animation import FuncAnimation
import time
from scipy.ndimage import binary_dilation, binary_erosion

class FieldSimulator:
    def __init__(self, width_meters=100, height_meters=100, image_size_meters=5, prediction_threshold=0.5):
        """
        Initialize field simulator
        width_meters: Width of field in meters
        height_meters: Height of field in meters
        image_size_meters: Size of each image capture in meters
        prediction_threshold: Confidence threshold for weed detection (default: 0.5)
        """
        self.width = width_meters
        self.height = height_meters
        self.image_size = image_size_meters
        self.prediction_threshold = prediction_threshold
        
        # Calculate grid dimensions
        self.grid_width = int(np.ceil(width_meters / image_size_meters))
        self.grid_height = int(np.ceil(height_meters / image_size_meters))
        
        # Initialize empty field
        self.field = np.zeros((self.grid_height, self.grid_width))
        self.weed_locations = []
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else "cpu"))
        self.model = load_model("models/RESNET/full_model.pth", device=device)
        self.model.eval()
        
        # Store predictions and confidences
        self.predictions = np.zeros((self.grid_height, self.grid_width))
        self.confidences = np.zeros((self.grid_height, self.grid_width))
        self.waypoints = []
        
        # Store drone path
        self.drone_path = []
        
        # Store actual field images
        self.field_images = {}  # (x,y) -> image tensor
    
    def get_field_images(self, field_id=None, min_images=50):
        """
        Get images from a specific field or find fields with enough images
        field_id: Specific field ID to use, or None to find suitable fields
        min_images: Minimum number of images required for a field to be considered
        """
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")
        image_files = os.listdir(os.path.join(test_dir, 'images/rgb'))
        
        # Group images by field ID and extract coordinates
        field_groups = {}
        field_coords = {}  # Store coordinates for each image
        field_weed_counts = {}  # Track number of images with weeds per field
        
        for img in image_files:
            # Parse filename: field_id_x1-y1-x2-y2.jpg
            parts = img.split('_')
            current_field_id = parts[0]
            coords = parts[1].split('.')[0].split('-')
            x1, y1, x2, y2 = map(int, coords)
            
            if current_field_id not in field_groups:
                field_groups[current_field_id] = []
                field_coords[current_field_id] = []
                field_weed_counts[current_field_id] = 0
            
            # Check if this image has a weed label
            weed_label_path = os.path.join(test_dir, 'labels/weed_cluster', img.replace('.jpg', '.png'))
            if os.path.exists(weed_label_path):
                # Load weed mask to check if it contains any weeds
                weed_mask = np.array(Image.open(weed_label_path))
                if np.any(weed_mask > 0):  # If there are any weed pixels
                    field_weed_counts[current_field_id] += 1
            
            field_groups[current_field_id].append(img)
            field_coords[current_field_id].append((x1, y1, x2, y2))
        
        # Find suitable fields with weeds
        suitable_fields = []
        for f, imgs in field_groups.items():
            if len(imgs) >= min_images and field_weed_counts[f] > 0:
                suitable_fields.append((f, len(imgs), field_weed_counts[f]))
        
        # Sort by number of weed images, then total images
        suitable_fields.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        if not suitable_fields:
            raise ValueError(f"No fields found with at least {min_images} images and weed clusters")
        
        # Print available fields with weed information
        print("\nAvailable fields with weeds:")
        for f, total_imgs, weed_imgs in suitable_fields[:5]:
            print(f"Field {f}: {total_imgs} total images, {weed_imgs} images with weeds")
        
        # Select field
        if field_id is None:
            # If no specific field requested, use the one with most weed images
            selected_field = suitable_fields[0][0]
        else:
            if field_id not in field_groups or len(field_groups[field_id]) < min_images:
                print(f"Warning: Requested field {field_id} not found or has too few images")
                selected_field = suitable_fields[0][0]
            else:
                selected_field = field_id
        
        images = field_groups[selected_field]
        coords = field_coords[selected_field]
        weed_count = field_weed_counts[selected_field]
        
        # Calculate field boundaries
        all_x1 = [c[0] for c in coords]
        all_y1 = [c[1] for c in coords]
        all_x2 = [c[2] for c in coords]
        all_y2 = [c[3] for c in coords]
        
        field_width = max(all_x2) - min(all_x1)
        field_height = max(all_y2) - min(all_y1)
        x_offset = min(all_x1)
        y_offset = min(all_y1)
        
        print(f"\nUsing field {selected_field} with {len(images)} total images and {weed_count} images containing weeds")
        print(f"Field dimensions: {field_width}x{field_height} pixels")
        print(f"Coordinate range: x[{min(all_x1)}-{max(all_x2)}], y[{min(all_y1)}-{max(all_y2)}]")
        
        return selected_field, images, coords, (x_offset, y_offset, field_width, field_height)
    
    def load_masks(self, image_name, test_dir):
        """
        Load mask and boundary information for an image
        """
        base_name = image_name.replace('.jpg', '.png')
        
        # Load weed cluster mask
        weed_path = os.path.join(test_dir, 'labels/weed_cluster', base_name)
        weed_mask = np.array(Image.open(weed_path)) if os.path.exists(weed_path) else None
        
        # Load field boundary mask
        boundary_path = os.path.join(test_dir, 'masks/boundary', base_name)
        boundary_mask = np.array(Image.open(boundary_path)) if os.path.exists(boundary_path) else None
        
        return weed_mask, boundary_mask
    
    def simulate_field(self, num_weed_clusters=10, cluster_size_range=(1, 3)):
        """
        Simulate weed clusters in the field
        num_weed_clusters: Number of weed clusters to generate
        cluster_size_range: (min_size, max_size) in grid cells
        """
        for _ in range(num_weed_clusters):
            # Random cluster center
            x = np.random.randint(0, self.grid_width)
            y = np.random.randint(0, self.grid_height)
            
            # Random cluster size
            size = np.random.randint(cluster_size_range[0], cluster_size_range[1] + 1)
            
            # Add cluster to field
            for dx in range(-size//2, size//2 + 1):
                for dy in range(-size//2, size//2 + 1):
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                        self.field[new_y, new_x] = 1
                        self.weed_locations.append((new_x * self.image_size, new_y * self.image_size))
    
    def process_images(self, num_images=400):
        """
        Process images from the dataset and map them to field locations
        """
        # Get images from a single field
        field_id, field_images = self.get_field_images(num_images)
        
        # Create custom dataset for these images
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")
        dataset = CustomDataset(test_dir)
        dataset.image_names = field_images
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Get device
        device = next(self.model.parameters()).device
        
        # Process each image
        print(f"Processing {len(field_images)} images...")
        with torch.no_grad():
            for i, (input, _) in enumerate(dataloader):
                # Get grid coordinates
                grid_x = i % self.grid_width
                grid_y = i // self.grid_width
                
                if grid_y >= self.grid_height:
                    break
                
                # Store image
                self.field_images[(grid_x, grid_y)] = input
                
                # Move input to same device as model
                input = input.to(device)
                
                # Get prediction
                output = self.model(input)
                confidence = output.item()
                prediction = (confidence >= self.prediction_threshold)
                
                # Store prediction and confidence
                self.predictions[grid_y, grid_x] = prediction
                self.confidences[grid_y, grid_x] = confidence
                
                # Update drone path
                self.drone_path.append((grid_x, grid_y))
                
                # If weed detected, add waypoint
                if prediction:
                    x_meters = grid_x * self.image_size + self.image_size/2
                    y_meters = grid_y * self.image_size + self.image_size/2
                    self.waypoints.append({
                        'x': float(x_meters),
                        'y': float(y_meters),
                        'confidence': float(confidence)
                    })
    
    def visualize(self, show_predictions=True, show_drone=True, show_confidence=True):
        """
        Visualize the field with actual weed locations, predictions, and drone path
        """
        if show_confidence:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
            axes = [ax1, ax2, ax3]
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            axes = [ax1, ax2]
        
        # Plot actual field
        ax1.imshow(self.field, cmap='Greens')
        ax1.set_title('Simulated Field')
        ax1.grid(True)
        
        # Plot predictions
        if show_predictions:
            ax2.imshow(self.predictions, cmap='Greens')
            ax2.set_title('Detected Weed Locations')
            ax2.grid(True)
        
        # Plot confidence heatmap
        if show_confidence:
            im = ax3.imshow(self.confidences, cmap='RdYlGn')
            ax3.set_title('Detection Confidence')
            ax3.grid(True)
            plt.colorbar(im, ax=ax3)
        
        # Add grid and labels
        for ax in axes:
            ax.set_xticks(np.arange(-0.5, self.grid_width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_height, 1), minor=True)
            ax.set_xticklabels([f'{i*self.image_size}m' for i in range(self.grid_width)])
            ax.set_yticklabels([f'{i*self.image_size}m' for i in range(self.grid_height)])
        
        # Plot drone path
        if show_drone and self.drone_path:
            path = np.array(self.drone_path)
            ax2.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5, label='Drone Path')
            ax2.plot(path[0, 0], path[0, 1], 'go', label='Start')
            ax2.plot(path[-1, 0], path[-1, 1], 'ro', label='End')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def animate_drone(self, interval=200):
        """
        Animate the drone moving across the field
        interval: Time between frames in milliseconds
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.predictions, cmap='Greens')
        ax.grid(True)
        
        # Initialize drone marker
        drone = Circle((0, 0), 0.2, color='red', label='Drone')
        ax.add_patch(drone)
        
        def update(frame):
            if frame < len(self.drone_path):
                x, y = self.drone_path[frame]
                drone.center = (x, y)
                ax.set_title(f'Drone Position: ({x*self.image_size}m, {y*self.image_size}m)')
            return drone,
        
        ani = FuncAnimation(fig, update, frames=len(self.drone_path),
                          interval=interval, blit=True, repeat=False)
        plt.show()
    
    def save_waypoints(self, filename='waypoints.json'):
        """
        Save detected weed locations as waypoints
        """
        with open(filename, 'w') as f:
            json.dump({
                'field_dimensions': {
                    'width': self.width,
                    'height': self.height,
                    'image_size': self.image_size
                },
                'waypoints': self.waypoints
            }, f, indent=4)
        print(f"Saved {len(self.waypoints)} waypoints to {filename}")
    
    def visualize_field_images(self, grid_size=(5, 5), start_pos=(0, 0)):
        """
        Display actual field images in a grid
        grid_size: (rows, cols) for display
        start_pos: (start_row, start_col) position in field to start displaying from
        """
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        
        for i in range(rows):
            for j in range(cols):
                field_x = start_pos[1] + j
                field_y = start_pos[0] + i
                
                if (field_x, field_y) in self.field_images:
                    # Get image tensor and convert to displayable format
                    img_tensor = self.field_images[(field_x, field_y)][0]  # Remove batch dimension
                    # Take first 3 channels (RGB) and convert to numpy
                    rgb_img = img_tensor[:3].permute(1, 2, 0).cpu().numpy()
                    
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    rgb_img = std * rgb_img + mean
                    rgb_img = np.clip(rgb_img, 0, 1)
                    
                    # Display image
                    axes[i, j].imshow(rgb_img)
                    
                    # Add prediction overlay
                    pred = self.predictions[field_y, field_x]
                    conf = self.confidences[field_y, field_x]
                    color = 'green' if pred else 'red'
                    axes[i, j].set_title(f'Conf: {conf:.2f}', color=color)
                else:
                    axes[i, j].imshow(np.zeros((512, 512, 3)))  # Black image for empty slots
                
                axes[i, j].axis('off')
        
        plt.suptitle(f'Field Images (Position {start_pos} to {(start_pos[0]+rows-1, start_pos[1]+cols-1)})')
        plt.tight_layout()
        plt.show()
    
    def visualize_stitched_field(self, field_id=None, show_predictions=True, show_ground_truth=True, 
                              show_masks=True, show_animation=False, show_waypoints=False):
        """
        Create and display a stitched view of the entire field with actual image positions
        field_id: Specific field to visualize, or None to use largest field
        show_predictions: Whether to show model predictions overlay
        show_ground_truth: Whether to show ground truth overlay
        show_masks: Whether to show weed mask overlay
        show_animation: Whether to show drone path animation (default: False)
        show_waypoints: Whether to show waypoint markers on the visualizations (default: False)
        """
        # Get field images and their coordinates
        field_id, field_images, coords, (x_offset, y_offset, field_width, field_height) = self.get_field_images(field_id)
        
        # Create empty arrays for RGB image and overlays
        stitched_rgb = np.zeros((field_height, field_width, 3))
        if show_predictions:
            pred_overlay = np.zeros((field_height, field_width, 4))
        if show_ground_truth:
            truth_overlay = np.zeros((field_height, field_width, 4))
        if show_masks:
            weed_mask_full = np.zeros((field_height, field_width))
        
        # Calculate rows and columns for a square layout
        n_plots = 1 + sum([show_predictions, show_ground_truth, show_masks])
        if n_plots <= 2:
            n_rows, n_cols = 1, n_plots
        else:
            n_rows, n_cols = 2, 2
        
        # Create figure with adjusted size and spacing
        fig = plt.figure(figsize=(n_cols*4.8, n_rows*4.8))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        # Create subplot grid
        axes = []
        for i in range(n_plots):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            axes.append(ax)
        
        titles = ['RGB Field View']
        if show_predictions:
            titles.append('Model Predictions')
        if show_ground_truth:
            titles.append('Ground Truth')
        if show_masks:
            titles.append('Weed Mask')
        
        # Denormalization parameters
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        print("Stitching images together...")
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")
        
        # Create dataset for loading images
        dataset = CustomDataset(test_dir)
        dataset.image_names = field_images
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Process each image
        device = next(self.model.parameters()).device
        waypoints = []
        
        for i, (input, label) in enumerate(dataloader):
            # Get coordinates for this image
            x1, y1, x2, y2 = coords[i]
            x1 -= x_offset
            x2 -= x_offset
            y1 -= y_offset
            y2 -= y_offset
            
            # Convert and denormalize image
            img = input[0, :3].permute(1, 2, 0).cpu().numpy()
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Add image to stitched result
            stitched_rgb[y1:y2, x1:x2] = img
            
            # Get model prediction
            input = input.to(device)
            with torch.no_grad():
                output = self.model(input)
                confidence = output.item()
                prediction = (confidence >= self.prediction_threshold)
            
            if show_predictions:
                if prediction:
                    pred_overlay[y1:y2, x1:x2] = [1, 0, 0, confidence * 0.3]
                else:
                    pred_overlay[y1:y2, x1:x2] = [0, 1, 0, (1-confidence) * 0.3]
            
            if show_ground_truth:
                truth = label.item()
                if truth:
                    truth_overlay[y1:y2, x1:x2] = [1, 0, 0, 0.3]
                else:
                    truth_overlay[y1:y2, x1:x2] = [0, 1, 0, 0.3]
            
            if show_masks:
                # Load actual weed mask
                mask_path = os.path.join(test_dir, 'labels/weed_cluster', 
                                       field_images[i].replace('.jpg', '.png'))
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path))
                    weed_mask_full[y1:y2, x1:x2] = mask
            
            # Add waypoint if weed detected
            if prediction:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                waypoints.append({
                    'x': float(center_x),
                    'y': float(center_y),
                    'confidence': float(confidence)
                })
        
        # Plot results
        for ax, title in zip(axes, titles):
            if title == 'RGB Field View':
                ax.imshow(stitched_rgb)
            elif title == 'Model Predictions':
                ax.imshow(stitched_rgb)
                ax.imshow(pred_overlay)
            elif title == 'Ground Truth':
                ax.imshow(stitched_rgb)
                ax.imshow(truth_overlay)
            elif title == 'Weed Mask':
                ax.imshow(weed_mask_full, cmap='hot', alpha=0.7)
                ax.imshow(stitched_rgb, alpha=0.3)  # Overlay RGB with low opacity
            
            # Add waypoints to all plots only if requested
            if show_waypoints and waypoints:
                waypoint_x = [w['x'] for w in waypoints]
                waypoint_y = [w['y'] for w in waypoints]
                ax.scatter(waypoint_x, waypoint_y, c='yellow', marker='x', s=50, label='Waypoints')
            
            ax.set_title(title)
            ax.axis('off')
        
        # Add legend only if showing waypoints
        if show_waypoints:
            legend_elements = [
                Patch(facecolor='red', alpha=0.3, label='Weed'),
                Patch(facecolor='green', alpha=0.3, label='No weed'),
                plt.Line2D([0], [0], marker='x', color='yellow', label='Waypoints',
                          markerfacecolor='yellow', markersize=10, linestyle='None')
            ]
        else:
            legend_elements = [
                Patch(facecolor='red', alpha=0.3, label='Weed'),
                Patch(facecolor='green', alpha=0.3, label='No weed')
            ]
        
        for ax in axes[1:]:
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle(f'Field {field_id}', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Save waypoints
        waypoints_file = f'waypoints_{field_id}.json'
        with open(waypoints_file, 'w') as f:
            json.dump({
                'field_id': field_id,
                'field_dimensions': {
                    'width': field_width,
                    'height': field_height
                },
                'waypoints': waypoints
            }, f, indent=4)
        print(f"\nSaved {len(waypoints)} waypoints to {waypoints_file}")
        
        if show_animation and waypoints:
            self.animate_drone_path(stitched_rgb, waypoints)
        
        # Print accuracy metrics
        if show_ground_truth:
            true_positives = np.sum(np.logical_and(pred_overlay[:,:,0] > 0, truth_overlay[:,:,0] > 0))
            true_negatives = np.sum(np.logical_and(pred_overlay[:,:,1] > 0, truth_overlay[:,:,1] > 0))
            false_positives = np.sum(np.logical_and(pred_overlay[:,:,0] > 0, truth_overlay[:,:,1] > 0))
            false_negatives = np.sum(np.logical_and(pred_overlay[:,:,1] > 0, truth_overlay[:,:,0] > 0))
            
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nModel Performance Metrics for Field {field_id}:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")
    
    def animate_drone_path(self, background, waypoints, interval=50):
        """
        Animate drone moving between waypoints
        """
        if not waypoints:
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(background)
        
        # Plot all waypoints
        waypoint_x = [w['x'] for w in waypoints]
        waypoint_y = [w['y'] for w in waypoints]
        ax.scatter(waypoint_x, waypoint_y, c='yellow', marker='x', s=50, label='Waypoints')
        
        # Create drone marker
        drone = plt.Circle((waypoint_x[0], waypoint_y[0]), 50, color='red', label='Drone')
        ax.add_patch(drone)
        
        def update(frame):
            if frame < len(waypoints):
                drone.center = (waypoint_x[frame], waypoint_y[frame])
            return [drone]
        
        ani = FuncAnimation(fig, update, frames=len(waypoints),
                          interval=interval, blit=True)
        
        ax.set_title('Drone Path Animation')
        ax.legend()
        plt.show()
    
    def create_synthetic_field(self, grid_width=10, grid_height=10, weed_ratio=0.3):
        """
        Create a synthetic field by sampling random images from the dataset
        grid_width: Number of images in width
        grid_height: Number of images in height
        weed_ratio: Approximate ratio of images that should contain weeds (0-1)
        """
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")
        
        # Get all available images and separate them into weed and non-weed images
        image_files = os.listdir(os.path.join(test_dir, 'images/rgb'))
        weed_images = []
        non_weed_images = []
        
        print("Categorizing available images...")
        for img in image_files:
            weed_label_path = os.path.join(test_dir, 'labels/weed_cluster', img.replace('.jpg', '.png'))
            if os.path.exists(weed_label_path):
                weed_mask = np.array(Image.open(weed_label_path))
                if np.any(weed_mask > 0):
                    weed_images.append(img)
                else:
                    non_weed_images.append(img)
            else:
                non_weed_images.append(img)
        
        print(f"Found {len(weed_images)} images with weeds and {len(non_weed_images)} without weeds")
        
        # Calculate how many weed images we need
        total_images = grid_width * grid_height
        target_weed_images = int(total_images * weed_ratio)
        target_non_weed_images = total_images - target_weed_images
        
        # Sample images randomly
        selected_weed_images = np.random.choice(weed_images, size=min(target_weed_images, len(weed_images)), replace=True)
        selected_non_weed_images = np.random.choice(non_weed_images, size=target_non_weed_images, replace=True)
        
        # Combine and shuffle
        all_selected_images = np.concatenate([selected_weed_images, selected_non_weed_images])
        np.random.shuffle(all_selected_images)
        
        # Create dataset for these images
        dataset = CustomDataset(test_dir)
        dataset.image_names = list(all_selected_images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Initialize arrays for visualization
        image_size = 512  # Assuming all images are 512x512
        stitched_rgb = np.zeros((grid_height * image_size, grid_width * image_size, 3))
        pred_overlay = np.zeros((grid_height * image_size, grid_width * image_size, 4))
        truth_overlay = np.zeros((grid_height * image_size, grid_width * image_size, 4))
        weed_mask_full = np.zeros((grid_height * image_size, grid_width * image_size))
        
        # Get device
        device = next(self.model.parameters()).device
        waypoints = []
        
        # Process each image
        print(f"\nCreating synthetic field ({grid_width}x{grid_height})...")
        with torch.no_grad():
            for i, (input, label) in enumerate(dataloader):
                if i >= total_images:
                    break
                
                # Calculate position in grid
                grid_x = i % grid_width
                grid_y = i // grid_width
                
                # Calculate pixel coordinates
                x1 = grid_x * image_size
                x2 = (grid_x + 1) * image_size
                y1 = grid_y * image_size
                y2 = (grid_y + 1) * image_size
                
                # Convert and denormalize image
                img = input[0, :3].permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Add image to stitched result
                stitched_rgb[y1:y2, x1:x2] = img
                
                # Get model prediction
                input = input.to(device)
                output = self.model(input)
                confidence = output.item()
                prediction = (confidence >= self.prediction_threshold)
                
                # Add prediction overlay
                if prediction:
                    pred_overlay[y1:y2, x1:x2] = [1, 0, 0, confidence * 0.3]
                else:
                    pred_overlay[y1:y2, x1:x2] = [0, 1, 0, (1-confidence) * 0.3]
                
                # Add ground truth overlay
                truth = label.item()
                if truth:
                    truth_overlay[y1:y2, x1:x2] = [1, 0, 0, 0.3]
                else:
                    truth_overlay[y1:y2, x1:x2] = [0, 1, 0, 0.3]
                
                # Add weed mask
                weed_label_path = os.path.join(test_dir, 'labels/weed_cluster', 
                                             dataset.image_names[i].replace('.jpg', '.png'))
                if os.path.exists(weed_label_path):
                    mask = np.array(Image.open(weed_label_path))
                    weed_mask_full[y1:y2, x1:x2] = mask
                
                # Add waypoint if weed detected
                if prediction:
                    center_x = x1 + image_size/2
                    center_y = y1 + image_size/2
                    waypoints.append({
                        'x': float(center_x),
                        'y': float(center_y),
                        'confidence': float(confidence)
                    })
        
        return stitched_rgb, pred_overlay, truth_overlay, weed_mask_full, waypoints
    
    def create_border_overlay(self, mask, color, border_width=2):
        """Helper function to create border overlay for a mask"""
        # Create border by subtracting eroded mask from dilated mask
        dilated = binary_dilation(mask, iterations=border_width)
        eroded = binary_erosion(mask, iterations=border_width)
        border = dilated & ~eroded
        
        # Create RGBA overlay
        overlay = np.zeros((*mask.shape, 4))
        overlay[border] = [*color, 1.0]  # Solid color for border
        return overlay
    
    def visualize_synthetic_field(self, grid_width=10, grid_height=10, weed_ratio=0.3,
                                show_predictions=True, show_ground_truth=True, 
                                show_masks=True, show_animation=False, show_waypoints=False,
                                show_evaluation=True, border_thickness=30):
        """
        Create and visualize a synthetic field
        """
        # Create synthetic field
        stitched_rgb, pred_overlay, truth_overlay, weed_mask_full, waypoints = \
            self.create_synthetic_field(grid_width, grid_height, weed_ratio)
        
        # Extract binary masks from overlays
        pred_mask = pred_overlay[:,:,0] > 0  # Red channel indicates weed prediction
        truth_mask = truth_overlay[:,:,0] > 0  # Red channel indicates actual weed
        
        # Define all masks for metrics
        tp_mask = pred_mask & truth_mask          # True Positive: predicted weed AND actually weed
        tn_mask = ~pred_mask & ~truth_mask        # True Negative: predicted no weed AND actually no weed
        fp_mask = pred_mask & ~truth_mask         # False Positive: predicted weed BUT actually no weed
        fn_mask = ~pred_mask & truth_mask         # False Negative: predicted no weed BUT actually weed
        
        # Calculate layout
        if show_evaluation:
            show_masks = False  # Replace mask with evaluation if enabled
        n_plots = 1 + sum([show_predictions, show_ground_truth, show_masks or show_evaluation])
        if n_plots <= 2:
            n_rows, n_cols = 1, n_plots
        else:
            n_rows, n_cols = 2, 2
        
        # Create figure with adjusted size
        fig = plt.figure(figsize=(n_cols*4.8, n_rows*4.8))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        # Create subplot grid
        axes = []
        for i in range(n_plots):
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            axes.append(ax)
        
        # Define plot titles
        titles = ['RGB Field View']
        if show_predictions:
            titles.append('Model Predictions')
        if show_ground_truth:
            titles.append('Ground Truth')
        if show_evaluation:
            titles.append('Evaluation Overlay')
        elif show_masks:
            titles.append('Weed Mask')
        
        # Plot results
        for ax, title in zip(axes, titles):
            # Always show the RGB image
            ax.imshow(stitched_rgb)
            
            if title == 'Model Predictions':
                pred_border = self.create_border_overlay(pred_mask, [1, 0, 0], border_width=border_thickness)
                ax.imshow(pred_border)
            
            elif title == 'Ground Truth':
                truth_border = self.create_border_overlay(truth_mask, [1, 0, 0], border_width=border_thickness)
                ax.imshow(truth_border)
            
            elif title == 'Evaluation Overlay':
                # Create evaluation overlay with borders
                eval_overlay = np.zeros((*stitched_rgb.shape[:2], 4))  # RGBA
                
                # True Positives (Green): Correctly identified weeds
                tp_overlay = self.create_border_overlay(tp_mask, [0, 1, 0], border_width=border_thickness)  # Green
                
                # False Positives (Yellow): Predicted weed but actually no weed
                fp_overlay = self.create_border_overlay(fp_mask, [1, 1, 0], border_width=border_thickness)  # Yellow
                
                # False Negatives (Blue): Missed actual weeds
                fn_overlay = self.create_border_overlay(fn_mask, [0, 0, 1], border_width=border_thickness)  # Blue
                
                # Combine overlays
                eval_overlay = np.maximum.reduce([tp_overlay, fp_overlay, fn_overlay])
                ax.imshow(eval_overlay)
                
                # Update legend for evaluation plot
                legend_elements = [
                    Patch(facecolor='green', label='Correct Detection'),
                    Patch(facecolor='yellow', label='False Positive'),
                    Patch(facecolor='blue', label='False Negative')
                ]
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.15), 
                         loc='upper right', ncol=3)
            
            elif title == 'Weed Mask':
                ax.imshow(weed_mask_full, cmap='hot', alpha=0.7)
                ax.imshow(stitched_rgb, alpha=0.3)
            
            # Add grid lines
            image_size = stitched_rgb.shape[0] // grid_height
            for i in range(grid_width + 1):
                ax.axvline(x=i * image_size, color='white', alpha=0.3, linestyle='--')
            for i in range(grid_height + 1):
                ax.axhline(y=i * image_size, color='white', alpha=0.3, linestyle='--')
            
            ax.set_title(title)
            ax.axis('off')
            
            # Add legend for non-evaluation plots
            if title != 'RGB Field View' and title != 'Evaluation Overlay':
                legend_elements = [Patch(facecolor='red', label='Weed Location')]
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.15), 
                         loc='upper right', ncol=1)
        
        plt.suptitle(f'Synthetic Field ({grid_width}x{grid_height}, {weed_ratio*100:.0f}% weed ratio)', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Calculate and print evaluation metrics
        if show_evaluation:
            tp = np.sum(tp_mask)
            tn = np.sum(tn_mask)
            fp = np.sum(fp_mask)
            fn = np.sum(fn_mask)
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print("\nEvaluation Metrics:")
            print(f"True Positives: {tp}")
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")
        
        # Save waypoints
        waypoints_file = f'waypoints_synthetic_{grid_width}x{grid_height}.json'
        with open(waypoints_file, 'w') as f:
            json.dump({
                'field_dimensions': {
                    'width': stitched_rgb.shape[1],
                    'height': stitched_rgb.shape[0],
                    'grid_width': grid_width,
                    'grid_height': grid_height,
                    'weed_ratio': weed_ratio
                },
                'waypoints': waypoints
            }, f, indent=4)
        print(f"\nSaved {len(waypoints)} waypoints to {waypoints_file}")


def main():
    # Create field simulator with custom prediction threshold
    print("Initializing field simulator...")
    simulator = FieldSimulator(
        width_meters=100, 
        height_meters=100, 
        image_size_meters=5,
        prediction_threshold=0.8
    )
    
    # # Example 1: Visualize real field
    # print("\nCreating real field view...")
    # simulator.visualize_stitched_field(
    #     field_id=None,
    #     show_predictions=True, 
    #     show_ground_truth=True,
    #     show_animation=False,
    #     show_waypoints=False
    # )
    
    # Example 2: Create synthetic field with evaluation overlay
    print("\nCreating synthetic field...")
    simulator.visualize_synthetic_field(
        grid_width=20,           # 8 images wide
        grid_height=16,          # 6 images tall
        weed_ratio=0.15,         # a percentage of images will contain weeds
        show_predictions=True,
        show_ground_truth=True,
        show_masks=False,       # Mask will be replaced by evaluation overlay
        show_animation=False,
        show_waypoints=False,
        show_evaluation=True,   # Show the evaluation overlay
        border_thickness=30
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main() 