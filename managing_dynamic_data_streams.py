# -*- coding: utf-8 -*-
"""Managing Dynamic Data Streams.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ChFTH0Of05AAZAxOv4ueVwBBcBRtwMYz
"""

#STEP 1

#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')


#Converting video into frames
import cv2
import numpy as np

url = '/content/drive/MyDrive/Copy of XVR_ch4_main_20230203230000_20230204000000.dav'
vidObj = cv2.VideoCapture(url)

count = 0

while True:
    success, image = vidObj.read()
    if success:
        cv2.imwrite(f"/content/drive/MyDrive/Duplicate/frame{count}.jpg", image)
    else:
        break
    count += 1
print(' Number of Frames generated :', count)

# Mount the Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import cv2
import os
import glob

# Specify the directory containing the .dav videos
video_folder = '/content/drive/MyDrive/'  # Replace with your folder path containing .dav videos

# Get a list of all .dav files in the specified directory
video_files = glob.glob(os.path.join(video_folder, '*.dav'))

# Loop through each video
for idx, video_path in enumerate(video_files):
    # Set the folder path to save frames for this video
    output_folder = f"/content/drive/MyDrive/Duplicate{idx + 1}"

    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    vidObj = cv2.VideoCapture(video_path)

    count = 0

    # Read and save frames
    while True:
        success, image = vidObj.read()
        if success:
            # Save each frame as a JPEG file in the specified folder
            cv2.imwrite(f"{output_folder}/frame{count}.jpg", image)
        else:
            break
        count += 1

    # Print the number of frames generated for this video
    print(f"Number of frames generated for video {idx + 1} in {output_folder}: {count}")

print("All videos processed.")

#STEP 3

#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

#Removes similar images from a folder in Google Drive and displays only the unique images

# Import necessary libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Define the path to the folder containing the images in your Google Drive
folder_path = '/content/drive/MyDrive/Duplicate/'

# Define a list to store the unique image files
unique_images = []

# Loop through all files in the folder and compare each image with the previous images
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        # Resize the image to a fixed size to make comparisons easier
        img = cv2.resize(img, (256, 256))
        # Convert the image to grayscale for comparison
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate the histogram of the grayscale image
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        # Calculate the Bhattacharyya distance between the histogram of the current image and the previous images
        similar = False
        for hist_ref in unique_images:
            similarity = cv2.compareHist(hist_ref, hist, cv2.HISTCMP_BHATTACHARYYA)
            if similarity < 0.07:
                # The current image is similar to a previous image, so skip it
                similar = True
                break
        if not similar:
            # The current image is unique, so add it to the list of unique images
            unique_images.append(hist)
            # Plot the image
            #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #plt.axis('off')
            plt.show()
#resizes each image to a fixed size, converts it to grayscale, calculates the histogram, normalizes the histogram, and compares it to the histograms of the previous images using the Bhattacharyya distance. If the current image is similar to a previous image, it is skipped. Otherwise, it is added to the list of unique images and displayed using matplotlib.pyplot.imshow(). The threshold for similarity can be adjusted by changing the value 0.5 in the if similarity < 0.5: line.
            save_path = '/content/drive/MyDrive/Unique/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = os.path.splitext(filename)[0] + '_unique.jpg'
            save_path = os.path.join(save_path, save_name)
            cv2.imwrite(save_path, img)

# Mount the Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import glob

# Define the base path where the Duplicate folders are located
base_folder_path = '/content/drive/MyDrive/'

# Get a list of all folders starting with "Duplicate"
duplicate_folders = sorted(glob.glob(os.path.join(base_folder_path, 'Duplicateeee*')))

# Loop through each Duplicate folder
for folder_idx, folder_path in enumerate(duplicate_folders, start=1):
    # Define the folder to save unique frames for this specific Duplicate folder
    save_folder = os.path.join(base_folder_path, f'UniqueFrames{folder_idx}')
    os.makedirs(save_folder, exist_ok=True)

    # Define a list to store the unique image histograms
    unique_images = []

    # Loop through all files in the current Duplicate folder and compare images
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            # Resize the image to a fixed size for comparison
            img = cv2.resize(img, (256, 256))
            # Convert the image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculate and normalize the histogram
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Check for similarity with previously stored unique images
            similar = False
            for hist_ref in unique_images:
                similarity = cv2.compareHist(hist_ref, hist, cv2.HISTCMP_BHATTACHARYYA)
                if similarity < 0.07:  # Adjust this threshold if needed
                    similar = True
                    break

            # If image is unique, add it to unique_images and save it
            if not similar:
                unique_images.append(hist)
                save_name = os.path.splitext(filename)[0] + '_unique.jpg'
                save_path = os.path.join(save_folder, save_name)
                cv2.imwrite(save_path, img)
                # Optionally, display the image
                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                # plt.show()

    print(f"Processed folder '{folder_path}', unique frames saved to '{save_folder}'.")

print("All folders processed.")

# Import necessary libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')


# Define the path to the folder containing the images in your Google Drive
folder_path = '/content/drive/MyDrive/Duplicate/'

# Define a list to store the unique image histograms
unique_images = []

# Loop through all files in the folder and compare each image with the previous images
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        # Resize the image to a fixed size to make comparisons easier
        img = cv2.resize(img, (256, 256))
        # Convert the image to grayscale for comparison
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate the histogram of the grayscale image
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        # Calculate the Bhattacharyya distance between the histogram of the current image and the previous images
        similar = False
        for hist_ref in unique_images:
            similarity = cv2.compareHist(hist_ref, hist, cv2.HISTCMP_BHATTACHARYYA)
            if similarity < 0.07:  # Consider images similar if distance is below this threshold
                similar = True
                break

        if not similar:
            # The current image is unique, so add it to the list of unique images
            unique_images.append(hist)

            # Plot the unique image and its histogram
            plt.figure(figsize=(12, 6))

            # Display the image
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Unique Image')
            plt.axis('off')

            # Plot the histogram
            plt.subplot(1, 2, 2)
            plt.plot(hist, color='gray')
            plt.title('Grayscale Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')

            # Show the plot
            plt.show()

            # Save the unique image to the output folder
            save_path = '/content/drive/MyDrive/Unique/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = os.path.splitext(filename)[0] + '_unique.jpg'
            save_path = os.path.join(save_path, save_name)
            cv2.imwrite(save_path, img)

# Import necessary libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

# Load the image
image_path = '/content/drive/MyDrive/Unique/'  # Change this to the path of your image
img = cv2.imread(image_path)

# Convert to grayscale (if you want a grayscale histogram)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate the grayscale histogram
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

# Normalize the histogram
hist = cv2.normalize(hist, hist).flatten()

# Plot the image and its histogram
plt.figure(figsize=(12, 6))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Plot the histogram
plt.subplot(1, 2, 2)
plt.plot(hist, color='gray')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Import necessary libraries
import os
import cv2
import matplotlib.pyplot as plt

#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

# Define the path to the folder containing the unique images
unique_frames_folder = '/content/drive/MyDrive/Unique/'  # Change this to your actual folder path

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(unique_frames_folder) if f.endswith('.jpg')]

# Loop through and display the histogram for each unique frame
plt.figure(figsize=(15, 15))  # Adjust the figure size
columns = 2  # Number of columns for displaying histograms

for i, filename in enumerate(image_files):
    img_path = os.path.join(unique_frames_folder, filename)
    img = cv2.imread(img_path)

    # Convert to grayscale for histogram visualization
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram for the grayscale image
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

    # Plot the image alongside its histogram
    plt.subplot(len(image_files) * 2 // columns + 1, columns, i * 2 + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Frame {i + 1}')
    plt.axis('off')

    # Plot the histogram
    plt.subplot(len(image_files) * 2 // columns + 1, columns, i * 2 + 2)
    plt.plot(hist)
    plt.title(f'Histogram of Frame {i + 1}')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

# Show the plot
#plt.tight_layout()
plt.show()

# Import necessary libraries
import os
import cv2
import matplotlib.pyplot as plt

#Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

# Define the path to the folder containing the unique images
unique_frames_folder = '/content/drive/MyDrive/Unique/'  # Change this to your actual folder path

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(unique_frames_folder) if f.endswith('.jpg')]

# Set the figure size for better visibility
plt.figure(figsize=(10, 5 * len(image_files)))  # Adjust figure size dynamically based on number of images
columns = 1  # Number of columns for displaying histograms

for i, filename in enumerate(image_files):
    img_path = os.path.join(unique_frames_folder, filename)
    img = cv2.imread(img_path)

    # Split the image into B, G, R channels
    channels = cv2.split(img)
    colors = ("b", "g", "r")  # Blue, Green, Red

    # Create a subplot for each image's histogram
    plt.subplot(len(image_files), columns, i + 1)

    # Plot histograms for each color channel
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.title(f'Color Histogram of Frame {i + 1}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

# Adjust the layout for better spacing between subplots
plt.subplots_adjust(hspace=0.5)  # Increase the space between plots vertically

# Show the plot
plt.show()

# Import necessary libraries
import os
import cv2
import matplotlib.pyplot as plt

# Define the path to the folder containing the unique images
unique_frames_folder = '/content/drive/MyDrive/Unique/'  # Change this to your actual folder path

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(unique_frames_folder) if f.endswith('.jpg')]

# Set the figure size for better visibility
plt.figure(figsize=(10, 5 * len(image_files)))  # Adjust figure size dynamically based on number of images
columns = 1  # Number of columns for displaying histograms

for i, filename in enumerate(image_files):
    img_path = os.path.join(unique_frames_folder, filename)
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a subplot for each image's histogram
    plt.subplot(len(image_files), columns, i + 1)

    # Plot grayscale histogram
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    plt.plot(hist, color='black')
    plt.xlim([0, 256])

    plt.title(f'Grayscale Histogram of Frame {i + 1}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

# Adjust the layout for better spacing between subplots
plt.subplots_adjust(hspace=0.5)  # Increase the space between plots vertically

# Show the plot
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the folder containing the unique images
unique_frames_folder = '/content/drive/MyDrive/Data/A1/Channel1/Frames1/'  # Change this to your actual folder path

# Initialize an array to store the accumulated histogram
accumulated_hist = np.zeros((256, 1), dtype=float)  # For grayscale images, we have 256 intensity levels

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(unique_frames_folder) if f.endswith('.jpg')]

for filename in image_files:
    img_path = os.path.join(unique_frames_folder, filename)
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram for the grayscale image
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    # Add the current image's histogram to the accumulated histogram
    accumulated_hist += hist

# Normalize the accumulated histogram for proper visualization
accumulated_hist = accumulated_hist / len(image_files)

# Plot the accumulated histogram
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist, color='black')
plt.title('Combined Grayscale Histogram of All Images')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])

# Display the plot
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the folder containing the unique images
unique_frames_folder = '/content/drive/MyDrive/Data/A1/Channel1/Frames1/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(unique_frames_folder) if f.endswith('.jpg')]

# Loop through each image file
for filename in image_files:
    img_path = os.path.join(unique_frames_folder, filename)
    img = cv2.imread(img_path)

    # Split the image into its color channels
    blue_channel, green_channel, red_channel = cv2.split(img)

    # Calculate the histogram for each color channel
    hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
    hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

    # Add the current image's histograms to the accumulated histograms
    accumulated_hist_red += hist_red
    accumulated_hist_green += hist_green
    accumulated_hist_blue += hist_blue

# Normalize the accumulated histograms for proper visualization
num_images = len(image_files)
accumulated_hist_red /= num_images
accumulated_hist_green /= num_images
accumulated_hist_blue /= num_images

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()

# Display the plot
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the parent folder containing multiple subfolders with images
parent_folder = '/content/drive/MyDrive/Data/A1/Channel1/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)

# Initialize a counter for total images
num_images = 0

# Loop through each subfolder and its files
for root, dirs, files in os.walk(parent_folder):
    for filename in files:
        if filename.endswith('.jpg'):  # Process only jpg files
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # Check if the image was loaded properly
            if img is not None:
                # Split the image into its color channels
                blue_channel, green_channel, red_channel = cv2.split(img)

                # Calculate the histogram for each color channel
                hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
                hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

                # Add the current image's histograms to the accumulated histograms
                accumulated_hist_red += hist_red
                accumulated_hist_green += hist_green
                accumulated_hist_blue += hist_blue

                # Increment the image counter
                num_images += 1

# Normalize the accumulated histograms for proper visualization
if num_images > 0:  # Prevent division by zero
    accumulated_hist_red /= num_images
    accumulated_hist_green /= num_images
    accumulated_hist_blue /= num_images
else:
    print("No images found in the specified folder.")

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()

# Display the plot
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the parent folder containing multiple subfolders with images
parent_folder = '/content/drive/MyDrive/Data/A1/Channel1/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)
accumulated_hist_gray = np.zeros((256, 1), dtype=float)

# Initialize a counter for total images
num_images = 0

# Loop through each subfolder and its files
for root, dirs, files in os.walk(parent_folder):
    for filename in files:
        if filename.endswith('.jpg'):  # Process only jpg files
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # Check if the image was loaded properly
            if img is not None:
                # Split the image into its color channels
                blue_channel, green_channel, red_channel = cv2.split(img)

                # Calculate the histogram for each color channel
                hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
                hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

                # Add the current image's histograms to the accumulated histograms
                accumulated_hist_red += hist_red
                accumulated_hist_green += hist_green
                accumulated_hist_blue += hist_blue

                # Convert the image to grayscale and calculate the histogram
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                accumulated_hist_gray += hist_gray

                # Increment the image counter
                num_images += 1

# Normalize the accumulated histograms for proper visualization
if num_images > 0:  # Prevent division by zero
    accumulated_hist_red /= num_images
    accumulated_hist_green /= num_images
    accumulated_hist_blue /= num_images
    accumulated_hist_gray /= num_images
else:
    print("No images found in the specified folder.")

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Plot the accumulated grayscale histogram
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_gray, color='gray', label='Grayscale Channel')
plt.title('Combined Grayscale Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the parent folder containing multiple subfolders with images
parent_folder = '/content/drive/MyDrive/Data/A2/Channel2/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)
accumulated_hist_gray = np.zeros((256, 1), dtype=float)

# Initialize a counter for total images
num_images = 0

# Loop through each subfolder and its files
for root, dirs, files in os.walk(parent_folder):
    for filename in files:
        if filename.endswith('.jpg'):  # Process only jpg files
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # Check if the image was loaded properly
            if img is not None:
                # Split the image into its color channels
                blue_channel, green_channel, red_channel = cv2.split(img)

                # Calculate the histogram for each color channel
                hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
                hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

                # Add the current image's histograms to the accumulated histograms
                accumulated_hist_red += hist_red
                accumulated_hist_green += hist_green
                accumulated_hist_blue += hist_blue

                # Convert the image to grayscale and calculate the histogram
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                accumulated_hist_gray += hist_gray

                # Increment the image counter
                num_images += 1

# Normalize the accumulated histograms for proper visualization
if num_images > 0:  # Prevent division by zero
    accumulated_hist_red /= num_images
    accumulated_hist_green /= num_images
    accumulated_hist_blue /= num_images
    accumulated_hist_gray /= num_images
else:
    print("No images found in the specified folder.")

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Plot the accumulated grayscale histogram
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_gray, color='gray', label='Grayscale Channel')
plt.title('Combined Grayscale Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the parent folder containing multiple subfolders with images
parent_folder = '/content/drive/MyDrive/Data/A3/Channel3/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)
accumulated_hist_gray = np.zeros((256, 1), dtype=float)

# Initialize a counter for total images
num_images = 0

# Loop through each subfolder and its files
for root, dirs, files in os.walk(parent_folder):
    for filename in files:
        if filename.endswith('.jpg'):  # Process only jpg files
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # Check if the image was loaded properly
            if img is not None:
                # Split the image into its color channels
                blue_channel, green_channel, red_channel = cv2.split(img)

                # Calculate the histogram for each color channel
                hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
                hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

                # Add the current image's histograms to the accumulated histograms
                accumulated_hist_red += hist_red
                accumulated_hist_green += hist_green
                accumulated_hist_blue += hist_blue

                # Convert the image to grayscale and calculate the histogram
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                accumulated_hist_gray += hist_gray

                # Increment the image counter
                num_images += 1

# Normalize the accumulated histograms for proper visualization
if num_images > 0:  # Prevent division by zero
    accumulated_hist_red /= num_images
    accumulated_hist_green /= num_images
    accumulated_hist_blue /= num_images
    accumulated_hist_gray /= num_images
else:
    print("No images found in the specified folder.")

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Plot the accumulated grayscale histogram
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_gray, color='gray', label='Grayscale Channel')
plt.title('Combined Grayscale Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the parent folder containing multiple subfolders with images
parent_folder = '/content/drive/MyDrive/Data/A4/Channel4/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)
accumulated_hist_gray = np.zeros((256, 1), dtype=float)

# Initialize a counter for total images
num_images = 0

# Loop through each subfolder and its files
for root, dirs, files in os.walk(parent_folder):
    for filename in files:
        if filename.endswith('.jpg'):  # Process only jpg files
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # Check if the image was loaded properly
            if img is not None:
                # Split the image into its color channels
                blue_channel, green_channel, red_channel = cv2.split(img)

                # Calculate the histogram for each color channel
                hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
                hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

                # Add the current image's histograms to the accumulated histograms
                accumulated_hist_red += hist_red
                accumulated_hist_green += hist_green
                accumulated_hist_blue += hist_blue

                # Convert the image to grayscale and calculate the histogram
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                accumulated_hist_gray += hist_gray

                # Increment the image counter
                num_images += 1

# Normalize the accumulated histograms for proper visualization
if num_images > 0:  # Prevent division by zero
    accumulated_hist_red /= num_images
    accumulated_hist_green /= num_images
    accumulated_hist_blue /= num_images
    accumulated_hist_gray /= num_images
else:
    print("No images found in the specified folder.")

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Plot the accumulated grayscale histogram
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_gray, color='gray', label='Grayscale Channel')
plt.title('Combined Grayscale Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the parent folder containing multiple subfolders with images
parent_folder = '/content/drive/MyDrive/Data/A5/Channel5/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)
accumulated_hist_gray = np.zeros((256, 1), dtype=float)

# Initialize a counter for total images
num_images = 0

# Loop through each subfolder and its files
for root, dirs, files in os.walk(parent_folder):
    for filename in files:
        if filename.endswith('.jpg'):  # Process only jpg files
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # Check if the image was loaded properly
            if img is not None:
                # Split the image into its color channels
                blue_channel, green_channel, red_channel = cv2.split(img)

                # Calculate the histogram for each color channel
                hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
                hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

                # Add the current image's histograms to the accumulated histograms
                accumulated_hist_red += hist_red
                accumulated_hist_green += hist_green
                accumulated_hist_blue += hist_blue

                # Convert the image to grayscale and calculate the histogram
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                accumulated_hist_gray += hist_gray

                # Increment the image counter
                num_images += 1

# Normalize the accumulated histograms for proper visualization
if num_images > 0:  # Prevent division by zero
    accumulated_hist_red /= num_images
    accumulated_hist_green /= num_images
    accumulated_hist_blue /= num_images
    accumulated_hist_gray /= num_images
else:
    print("No images found in the specified folder.")

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Plot the accumulated grayscale histogram
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_gray, color='gray', label='Grayscale Channel')
plt.title('Combined Grayscale Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the parent folder containing multiple subfolders with images
parent_folder = '/content/drive/MyDrive/Data/A6/Channel6/'  # Change this to your actual folder path

# Initialize arrays to store the accumulated histograms for each channel
accumulated_hist_red = np.zeros((256, 1), dtype=float)
accumulated_hist_green = np.zeros((256, 1), dtype=float)
accumulated_hist_blue = np.zeros((256, 1), dtype=float)
accumulated_hist_gray = np.zeros((256, 1), dtype=float)

# Initialize a counter for total images
num_images = 0

# Loop through each subfolder and its files
for root, dirs, files in os.walk(parent_folder):
    for filename in files:
        if filename.endswith('.jpg'):  # Process only jpg files
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            # Check if the image was loaded properly
            if img is not None:
                # Split the image into its color channels
                blue_channel, green_channel, red_channel = cv2.split(img)

                # Calculate the histogram for each color channel
                hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
                hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
                hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])

                # Add the current image's histograms to the accumulated histograms
                accumulated_hist_red += hist_red
                accumulated_hist_green += hist_green
                accumulated_hist_blue += hist_blue

                # Convert the image to grayscale and calculate the histogram
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
                accumulated_hist_gray += hist_gray

                # Increment the image counter
                num_images += 1

# Normalize the accumulated histograms for proper visualization
if num_images > 0:  # Prevent division by zero
    accumulated_hist_red /= num_images
    accumulated_hist_green /= num_images
    accumulated_hist_blue /= num_images
    accumulated_hist_gray /= num_images
else:
    print("No images found in the specified folder.")

# Plot the accumulated RGB histograms
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_red, color='red', label='Red Channel')
plt.plot(accumulated_hist_green, color='green', label='Green Channel')
plt.plot(accumulated_hist_blue, color='blue', label='Blue Channel')
plt.title('Combined RGB Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Plot the accumulated grayscale histogram
plt.figure(figsize=(10, 6))
plt.plot(accumulated_hist_gray, color='gray', label='Grayscale Channel')
plt.title('Combined Grayscale Histogram of All Images in Subfolders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.legend()
plt.show()