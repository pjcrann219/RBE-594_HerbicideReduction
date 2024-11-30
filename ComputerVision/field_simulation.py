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

class FieldSimulator:
    def __init__(self, width_meters=100, height_meters=100, image_size_meters=5):
        """
        Initialize field simulator
        width_meters: Width of field in meters
        height_meters: Height of field in meters
        image_size_meters: Size of each image capture in meters
        """
        self.width = width_meters
        self.height = height_meters
        self.image_size = image_size_meters
        
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
    
    def get_field_images(self, num_images=400):
        """
        Get images from the same field in the dataset and their actual positions
        """
        test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Agriculture-Vision-2021/val")
        image_files = os.listdir(os.path.join(test_dir, 'images/rgb'))
        
        # Group images by field ID and extract coordinates
        field_groups = {}
        field_coords = {}  # Store coordinates for each image
        for img in image_files:
            # Parse filename: field_id_x1-y1-x2-y2.jpg
            parts = img.split('_')
            field_id = parts[0]
            coords = parts[1].split('.')[0].split('-')
            x1, y1, x2, y2 = map(int, coords)
            
            if field_id not in field_groups:
                field_groups[field_id] = []
                field_coords[field_id] = []
            field_groups[field_id].append(img)
            field_coords[field_id].append((x1, y1, x2, y2))
        
        # Find field with most contiguous images
        field_counts = [(field, len(images)) for field, images in field_groups.items()]
        field_counts.sort(key=lambda x: x[1], reverse=True)
        
        if not field_counts:
            raise ValueError("No fields found in the dataset")
        
        # Select field with most images
        selected_field = field_counts[0][0]
        images = field_groups[selected_field]
        coords = field_coords[selected_field]
        
        # Calculate field boundaries
        all_x1 = [c[0] for c in coords]
        all_y1 = [c[1] for c in coords]
        all_x2 = [c[2] for c in coords]
        all_y2 = [c[3] for c in coords]
        
        field_width = max(all_x2) - min(all_x1)
        field_height = max(all_y2) - min(all_y1)
        x_offset = min(all_x1)
        y_offset = min(all_y1)
        
        print(f"Using field {selected_field} with {len(images)} images")
        print(f"Field dimensions: {field_width}x{field_height} pixels")
        print(f"Coordinate range: x[{min(all_x1)}-{max(all_x2)}], y[{min(all_y1)}-{max(all_y2)}]")
        
        return selected_field, images, coords, (x_offset, y_offset, field_width, field_height)
    
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
                prediction = (confidence >= 0.5)
                
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
    
    def visualize_stitched_field(self, show_predictions=True, show_ground_truth=True):
        """
        Create and display a stitched view of the entire field with actual image positions
        """
        # Get field images and their coordinates
        field_id, field_images, coords, (x_offset, y_offset, field_width, field_height) = self.get_field_images()
        
        # Create empty arrays for RGB image and overlays
        stitched_rgb = np.zeros((field_height, field_width, 3))
        if show_predictions:
            pred_overlay = np.zeros((field_height, field_width, 4))  # RGBA for overlay
        if show_ground_truth:
            truth_overlay = np.zeros((field_height, field_width, 4))  # RGBA for overlay
        
        # Create figure with subplots
        if show_ground_truth:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
            axes = [ax1, ax2, ax3]
            titles = ['RGB Field View', 'Model Predictions', 'Ground Truth']
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            axes = [ax1, ax2]
            titles = ['RGB Field View', 'Model Predictions']
        
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
                prediction = (confidence >= 0.5)
            
            if show_predictions:
                # Add prediction overlay
                if prediction:
                    pred_overlay[y1:y2, x1:x2] = [1, 0, 0, confidence * 0.3]  # Red for weed
                else:
                    pred_overlay[y1:y2, x1:x2] = [0, 1, 0, (1-confidence) * 0.3]  # Green for no weed
            
            if show_ground_truth:
                # Add ground truth overlay
                truth = label.item()
                if truth:
                    truth_overlay[y1:y2, x1:x2] = [1, 0, 0, 0.3]  # Red for weed
                else:
                    truth_overlay[y1:y2, x1:x2] = [0, 1, 0, 0.3]  # Green for no weed
        
        # Plot results
        for ax, title in zip(axes, titles):
            ax.imshow(stitched_rgb)
            if title == 'Model Predictions' and show_predictions:
                ax.imshow(pred_overlay)
            elif title == 'Ground Truth' and show_ground_truth:
                ax.imshow(truth_overlay)
            ax.set_title(title)
            ax.axis('off')
        
        # Add legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Weed'),
            Patch(facecolor='green', alpha=0.3, label='No weed')
        ]
        for ax in axes[1:]:
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
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
            
            print("\nModel Performance Metrics:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")


def main():
    # Create field simulator
    print("Initializing field simulator...")
    simulator = FieldSimulator(width_meters=100, height_meters=100, image_size_meters=5)
    
    # Show stitched field view with ground truth comparison
    print("\nCreating stitched field view...")
    simulator.visualize_stitched_field(show_predictions=True, show_ground_truth=True)
    
    # Save waypoints
    print("\nSaving waypoints...")
    simulator.save_waypoints()
    
    print("\nDone! Check waypoints.json for the detected weed locations.")


if __name__ == "__main__":
    main() 