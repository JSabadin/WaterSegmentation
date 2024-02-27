from dataloader import load_data, show_images, denormalize, load_whole_data
from network import SegmentationCNN
import torch
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import json
import cv2
import numpy as np
from ultralytics import YOLO 

# Datasets mean and std for denormalization. Change these values if you used different ones.
global mean, std
mean = [0.39206787943840027, 0.4307292103767395, 0.39849820733070374]
std = [0.2781929075717926, 0.28806477785110474, 0.31973788142204285]

def calculate_iou(pred, gt):
    """
    Calculate the Intersection over Union (IoU) of the predicted binary mask with the ground truth.
    """
    # pred and gt are binary tensors with shape [batch_size, ...]
    intersection = torch.logical_and(pred, gt)
    union = torch.logical_or(pred, gt)
    iou = torch.sum(intersection, dim=[1, 2, 3]) / torch.sum(union, dim=[1, 2, 3])
    return iou.mean()

def calculate_iou_bbox(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of the predicted bounding boxes with the ground truth. Used for YOLO detection.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def plot_metrics(train_losses, val_losses, train_ious, val_ious, num_epochs):
    """
    Plot the training and validation metrics over the epochs.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs', fontsize=14, fontstyle='italic')
    plt.ylabel('Loss', fontsize=14, fontstyle='italic')
    plt.title('Training and Validation Loss Over Epochs', fontsize=16, fontstyle='italic')
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_ious, label='Train IoU')
    plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU', linestyle='--')
    plt.xlabel('Epochs', fontsize=14, fontstyle='italic')
    plt.ylabel('IoU', fontsize=14, fontstyle='italic')
    plt.title('IoU Over Epochs', fontsize=16, fontstyle='italic')
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()


def show_predicted_images(test_loader, model, device, num_images=4):
    """
    Show a subset of images from the test set with their predicted masks.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i_batch, input in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, true_masks = input[0].to(device), input[1].to(device)

            # Forward pass to get the outputs from the model
            outputs = model(images)

            # Convert outputs to binary masks
            predicted_masks = torch.sigmoid(outputs) > 0.5

            # Convert predicted masks from logits to binary tensor for visualization
            predicted_masks_binary = predicted_masks.type(torch.float32)

            # Now call your show_images function with a subset of images and masks
            show_images(images, predicted_masks_binary, num_images=num_images,mean=mean,std=std,show_normalised=False)
            show_images(images, true_masks, num_images=num_images,mean=mean,std=std,show_normalised=False)
            break  # Only process the first batch


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint file.
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        train_ious = checkpoint['train_ious']
        val_losses = checkpoint['val_losses']
        val_ious = checkpoint['val_ious']
        print(f"Loaded checkpoint from epoch {epoch}.")
        return epoch, train_losses, train_ious, val_losses, val_ious
    return 0, [], [], [], []

def save_checkpoint(checkpoint_path, model, optimizer, epoch, train_losses, train_ious, val_losses, val_ious):
    """
    Save the model and optimizer state to a checkpoint file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_ious': train_ious,
        'val_losses': val_losses,
        'val_ious': val_ious
    }, checkpoint_path)

def log_results(log_file, epoch, train_loss, train_iou, val_loss, val_iou):
    """
    Log the training and validation results to a file in JSON format.
    """
    log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_iou': train_iou,
        'val_loss': val_loss,
        'val_iou': val_iou
    }
    with open(log_file, 'a') as file:
        file.write(json.dumps(log_data) + '\n')

def read_log_file(log_file):
    """
    Read the log file and return the data as a list of dictionaries.
    """
    with open(log_file, 'r') as file:
        log_data = [json.loads(line) for line in file]
    return log_data

def plot_from_log(log_file):
    """
    Plot the training and validation metrics from the log file.
    """
    log_data = read_log_file(log_file)
    epochs = [entry['epoch'] for entry in log_data]
    train_losses = [entry['train_loss'] for entry in log_data]
    train_ious = [entry['train_iou'] for entry in log_data]
    val_losses = [entry['val_loss'] for entry in log_data]
    val_ious = [entry['val_iou'] for entry in log_data]

    plot_metrics(train_losses, val_losses, train_ious, val_ious, len(epochs))

def train_and_validate(model, train_loader, val_loader, optimizer, l_ce, device, num_epochs, log_file, checkpoint_path):
    """
    Train and validate the model for a specified number of epochs.
    """
    start_epoch, train_losses, train_ious, val_losses, val_ious = load_checkpoint(checkpoint_path, model, optimizer)
    best_iou = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        avg_loss = 0
        total_train_iou = 0

        # Training loop
        for i_batch, input in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, masks = input[0].to(device), input[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = l_ce(outputs, masks)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            predicted_masks = torch.sigmoid(outputs) > 0.5
            total_train_iou += calculate_iou(predicted_masks, masks).item()

        avg_loss /= len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)

        # Validation loop
        model.eval()
        total_val_iou = 0
        total_val_loss = 0
        with torch.no_grad():
            for i_batch, input in tqdm(enumerate(val_loader), total=len(val_loader)):
                images, masks = input[0].to(device), input[1].to(device)
                outputs = model(images)
                loss = l_ce(outputs, masks)
                total_val_loss += loss.item()
                predicted_masks = torch.sigmoid(outputs) > 0.5
                total_val_iou += calculate_iou(predicted_masks, masks).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)

        # Log results and save checkpoint if better IoU is found
        log_results(log_file, epoch + 1, avg_loss, avg_train_iou, avg_val_loss, avg_val_iou)
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            print(f'Best weights saved to {checkpoint_path}')
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, train_losses, train_ious, val_losses, val_ious)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss: {avg_loss:.4f}, Average Train IoU: {avg_train_iou:.4f}, Average Validation Loss: {avg_val_loss:.4f}, Average Validation IoU: {avg_val_iou:.4f}")

def test(model, test_loader, device):
    """
    Test the model on the test set and print the average loss and IoU.
    """
    # Validation loop
    model.eval()
    total_test_iou = 0
    total_test_loss = 0
    with torch.no_grad():
        for i_batch, input in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, masks = input[0].to(device), input[1].to(device)
            outputs = model(images)
            loss = l_ce(outputs, masks)
            total_test_loss += loss.item()
            predicted_masks = torch.sigmoid(outputs) > 0.5
            total_test_iou += calculate_iou(predicted_masks, masks).item()

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_iou = total_test_iou / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}, Average Test IoU: {avg_test_iou:.4f}")


# def detect_obstacles(YOLO_model, loader, imagesize, plot_results=False):
#     """
#     Detect obstacles using the trained model and YOLO. Calculate the true positives, false positives, and false negatives using U-Net's ground truth and YOLO.
#     """
#     true_positives = false_negatives = false_positives = 0
#     for i_batch, item in enumerate(loader):
#         images, masks, paths, _ = item
#         for i in range(len(images)):
#             binary_image = (masks[i].cpu().detach().numpy() * 255.0).astype(np.uint8)
#             img = np.ascontiguousarray(denormalize(images[i], mean, std))
#             img_with_countours = img.copy() 
#             binary_image = binary_image.squeeze(0)  # Remove channel dimension
#             binary_image_3channel = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
#             contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#             boxes_of_interest = []
#             for contour in contours:
#                 mask = np.zeros(binary_image.shape, dtype=np.uint8)
#                 cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill the contour
                
#                 # Extract the pixels within the filled contour
#                 pixels_inside_contour = binary_image[mask == 255]
                
#                 # Calculate the percentage of black pixels inside the contour
#                 percentage_black = np.sum(pixels_inside_contour == 0) / pixels_inside_contour.size
#                 with_background_flag = True      
#                 # Calculate the percentage of black pixels inside the contour
#                 x, y, w, h = cv2.boundingRect(contour)
#                 # Check if at least 90 percent of the pixels inside the contour are black
#                 area = w*h
#                 if (x <= 0) or (y <= 0) or ((x + w) == imagesize[1] - 1) or ((y + h) == imagesize[0] - 1):
#                     # The contour touches the border
#                     with_background_flag = False
#                 else:
#                     with_background_flag = True
#                 if  (area > 25 and area < 46_501)  and with_background_flag and (percentage_black > 0.5):
#                     boxes_of_interest.append((x,y,x+w,y+h))
#                     print(x,y,(x + w),(y + h))
#                     cv2.rectangle(binary_image_3channel, (x, y), (x+w, y+h), (0, 255, 0), 1)
#                     cv2.rectangle(img_with_countours, (x, y), (x+w, y+h), (0, 255, 0), 1)

#             # CALCULATE TP FP FN
#             if not(len(boxes_of_interest) == 0):
#                 # YOLO DETECTION
#                 # Erode and then dilate the binary image
#                 yolo_mask = (masks[i].cpu().detach().numpy()).squeeze(0)
#                 kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
#                 # Majhno odpiranje
#                 yolo_mask = cv2.dilate(yolo_mask, kernel, iterations=2)
#                 yolo_mask = cv2.erode(yolo_mask, kernel, iterations=4)
#                 yolo_mask = cv2.dilate(yolo_mask, kernel, iterations=4)
#                 # Zapiranje
#                 yolo_mask = cv2.dilate(yolo_mask, kernel, iterations=5)
#                 yolo_mask = cv2.erode(yolo_mask, kernel, iterations=8)
#                 yolo_mask_3channel = np.stack((yolo_mask, yolo_mask, yolo_mask), axis=-1)
#                 yolo_img = np.clip(img * yolo_mask_3channel, 0, 255).astype(np.uint8)
#                 # # Evaluacija preko originalnih slik
#                 # yolo_img = img
#                 yolo_info = YOLO_model(yolo_img)

#                 for r in yolo_info: # Can be in batch
#                     detections = r.boxes.xyxy.cpu().numpy().tolist()

#                 for detec in detections:
#                     cv2.rectangle(img_with_countours, (int(detec[0]), int(detec[1])), (int(detec[2]), int(detec[3])), (255, 0, 0), 1)

#                 matched_detections = set()
#                 matched_boxes = set()
#                 true_positives_per_image = 0
#                 for i, box_oi in enumerate(boxes_of_interest):
#                     best_iou = 0
#                     best_det_idx = None
#                     for j, det in enumerate(detections):
#                         iou = calculate_iou_bbox(det, box_oi)
#                         if iou > best_iou:
#                             best_iou = iou
#                             best_det_idx = j
#                     if best_iou > 0.5:  # Assuming some IoU threshold, e.g., 0.5
#                         true_positives_per_image += 1
#                         matched_detections.add(best_det_idx)
#                         matched_boxes.add(i)
#                         print("TRUE POSITIVE")

#                 false_positives += len(detections) - len(matched_detections)
#                 false_negatives += len(boxes_of_interest) - len(matched_boxes)
#                 true_positives += true_positives_per_image

#                 if plot_results:
#                     plt.figure(figsize=(12, 6))

#                     # Display the first image
#                     plt.subplot(1, 3, 2)
#                     plt.imshow(img_with_countours)
#                     plt.axis('off')

#                     # Display the second image
#                     plt.subplot(1, 3, 1)
#                     plt.imshow(cv2.cvtColor(binary_image_3channel, cv2.COLOR_BGR2RGB))
#                     plt.axis('off')

#                     plt.subplot(1, 3, 3)
#                     plt.imshow(yolo_img)
#                     plt.axis('off')

#                     # Adjust subplot parameters to make the images touch each other
#                     plt.subplots_adjust(wspace=0, hspace=0)

#                     plt.savefig("figure", dpi =300, pad_inches=0.0, bbox_inches='tight')
#                     plt.show()



#     print(f"True positives: {true_positives}, False positives: {false_positives}, False negatives: {false_negatives}")
#     recall = true_positives / (true_positives + false_negatives)
#     precision = true_positives / (true_positives + false_positives)
#     F1 = 2 * (precision * recall) / (precision + recall)
#     print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {F1:.4f}")


def validate_detection_based_on_water(detec, mask, border_thickness=5, significance_threshold=0.5):
    """
    Validates if a detected object, based on its bounding box (`detec`),
    has a significant portion of its border intersecting with water, as defined by `mask`.
    
    Parameters:
    - detec: A list or tuple containing the bounding box coordinates [x1, y1, x2, y2].
    - mask: A binary mask where water is represented by 1s and non-water by 0s.
    - border_thickness: The thickness of the border to consider around the detection.
    - significance_threshold: The minimum percentage of the border that must intersect with water.
    
    Returns:
    - A boolean value indicating whether the detection passes the water intersection criterion.
    """
    # Initialize the background flag
    with_background_flag = True

    # Initialize an empty mask for the detected object's border
    border_mask = np.zeros_like(mask, dtype=np.uint8)

    # Extract bounding box coordinates and adjust to ensure they are within image boundaries
    x1, y1, x2, y2 = map(int, detec)
    offset  = 1

    x1 = x1 - border_thickness - offset
    y1 = y1 - border_thickness - offset
    x2 = x2 + border_thickness + offset
    y2 = y2 + border_thickness + offset


    # Draw the border on the mask
    border_mask[max(y1, 0):min(y1 + border_thickness, mask.shape[0]), max(x1, 0):min(x2, mask.shape[1])] = 1
    border_mask[max(y2 - border_thickness, 0):min(y2, mask.shape[0]), max(x1, 0):min(x2, mask.shape[1])] = 1
    border_mask[max(y1, 0):min(y2, mask.shape[0]), max(x1, 0):min(x1 + border_thickness, mask.shape[1])] = 1
    border_mask[max(y1, 0):min(y2, mask.shape[0]), max(x2 - border_thickness, 0):min(x2, mask.shape[1])] = 1

    # Calculate the intersection with the water mask
    intersection = border_mask * mask

    # Calculate the percentage of the border that intersects with water
    water_border_percentage = np.sum(intersection) / np.sum(border_mask)


    if (int(round(x1)) <= 0) or (int(round(y1)) <= 0) or (int(round(x2)) >= imagesize[1]) or (int(round(y2)) >= imagesize[0]):
        # The contour touches the border
        with_background_flag = False

    # Return True if the percentage meets or exceeds the significance threshold
    return water_border_percentage >= significance_threshold and with_background_flag

def detect_obstacles(YOLO_model, loader, imagesize, plot_results=False):
    """
    Detect obstacles using the trained model and YOLO. Calculate the true positives, false positives, and false negatives using U-Net's ground truth and YOLO.
    """
    true_positives = false_negatives = false_positives = 0
    for i_batch, item in enumerate(loader):
        images, masks, paths, _ = item
        for i in range(len(images)):
            binary_image = (masks[i].cpu().detach().numpy() * 255.0).astype(np.uint8)
            img = np.ascontiguousarray(denormalize(images[i], mean, std))
            img_with_countours = img.copy() 
            binary_image = binary_image.squeeze(0)  # Remove channel dimension
            binary_image_3channel = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            boxes_of_interest = []
            for contour in contours:
                mask = np.zeros(binary_image.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # Fill the contour
                
                # Extract the pixels within the filled contour
                pixels_inside_contour = binary_image[mask == 255]
                
                # Calculate the percentage of black pixels inside the contour
                percentage_black = np.sum(pixels_inside_contour == 0) / pixels_inside_contour.size
                with_background_flag = True      
                # Calculate the percentage of black pixels inside the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Check if at least 90 percent of the pixels inside the contour are black
                area = w*h
                if (int(round(x)) <= 0) or (int(round(y)) <= 0) or ((int(round(x + w))) >= imagesize[1]) or ((int(round(y + h))) >= imagesize[0]):
                    # The contour touches the border
                    with_background_flag = False
                else:
                    with_background_flag = True
                if  (area > 25 and area < 46_501)  and with_background_flag and (percentage_black > 0.5):
                    boxes_of_interest.append((x,y,x+w,y+h))
                    print(x,y,(x + w),(y + h))
                    cv2.rectangle(binary_image_3channel, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.rectangle(img_with_countours, (x, y), (x+w, y+h), (0, 255, 0), 1)

            if not(len(boxes_of_interest) == 0):
                # Assume 'mask' is your water mask with 1 for water and 0 for non-water
                mask = (masks[i].cpu().detach().numpy()).squeeze(0)
                
                # YOLO DETECTION
                yolo_info = YOLO_model(img)

                valid_detections = []  # List to store indices of valid detections (based on water border)
                for r in yolo_info:  # Process detections
                    detections = r.boxes.xyxy.cpu().numpy().tolist()

                    for j, det in enumerate(detections):
                        # Validate each detection based on water intersection
                        valid = validate_detection_based_on_water(det, mask, significance_threshold=0.84, border_thickness=1)
                        # valid = validate_detection_based_on_water(det, mask, significance_threshold=0.0, border_thickness=1)
                        if valid:
                            valid_detections.append(j)  # Add index of valid detection

                matched_detections = set()
                matched_boxes = set()
                true_positives_per_image = 0
                for i, box_oi in enumerate(boxes_of_interest):
                    best_iou = 0
                    best_det_idx = None
                    for j in valid_detections:  # Only consider valid detections
                    
                        det = detections[j]
                        print(f"Det: {det}")
                        x1, y1, x2, y2 = map(int, det)
                        cv2.rectangle(img_with_countours, (x1, y1), (x2,y2), (0, 0, 255), 1)
                        iou = calculate_iou_bbox(det, box_oi)
                        if iou > best_iou:
                            best_iou = iou
                            best_det_idx = j
                            cv2.rectangle(img_with_countours, (x1, y1), (x2,y2), (255, 0, 0), 1)
                    
                    if best_iou > 0.5:  # Assuming some IoU threshold, e.g., 0.5
                        true_positives_per_image += 1
                        matched_detections.add(best_det_idx)
                        matched_boxes.add(i)
                        print("TRUE POSITIVE")

                false_positives += len(valid_detections) - len(matched_detections)
                false_negatives += len(boxes_of_interest) - len(matched_boxes)
                true_positives += true_positives_per_image
                if len(valid_detections) - len(matched_detections) > 0:
                    print(f"False positives: {len(valid_detections) - len(matched_detections)}")
                    plot_results = True
                else:
                    plot_results = False
                if plot_results:
                    plt.figure(figsize=(12, 6))

                    # Display the first image
                    plt.subplot(1, 3, 2)
                    plt.imshow(img_with_countours)
                    plt.axis('off')

                    # Display the second image
                    plt.subplot(1, 3, 1)
                    plt.imshow(cv2.cvtColor(binary_image_3channel, cv2.COLOR_BGR2RGB))
                    plt.axis('off')

                    # Adjust subplot parameters to make the images touch each other
                    plt.subplots_adjust(wspace=0, hspace=0)

                    plt.savefig("figure", dpi =300, pad_inches=0.0, bbox_inches='tight')
                    plt.show()



    print(f"True positives: {true_positives}, False positives: {false_positives}, False negatives: {false_negatives}")
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    F1 = 2 * (precision * recall) / (precision + recall)
    print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {F1:.4f}")




if __name__ == "__main__":
    imagesize = (384, 512)
    batchsize = 2
    num_epochs = 30

    # Load training, validation, and test data
    train_loader, val_loader, test_loader = load_data(batchsize, imagesize)
    
    device = "cuda:0"
    model = SegmentationCNN().to(device)

    # Initialize loss function and optimizer
    l_ce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    # Log file path
    log_file = './logs/training_log2.json'

    # Check if checkpoint exists and load it
    checkpoint_path = "./checkpoints/checkpoint2.pth"

    # Train and validate the model
    train_and_validate(model, train_loader, val_loader, optimizer, l_ce, device, num_epochs, log_file, checkpoint_path) 
   
    # Plot the training and validation metrics from the log file
    plot_from_log(log_file)

    # Test the model after training
    test(model, test_loader, device)

    # Plot some predicted images
    show_predicted_images(test_loader, model, device)

    # Detect obstacles using the trained model and YOLO
    yolo = YOLO('yolov8n.pt')
    whole_loader = load_whole_data(batchsize, imagesize)
    detect_obstacles(yolo, whole_loader, imagesize, plot_results=False)
