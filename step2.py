import cv2 as cv
import numpy as np
import os

# texture distance

# Set the directory where the images are located
image_dir = './data/images/'

# Set the number of bits for each channel
bits = 6  # Change this value to find the "Goldilocks" choice

# Set the number of bins for each channel
bins = 2 ** bits

# Load the crowdsource data
crowd_data = np.loadtxt('./data/Crowd.txt', dtype = np.int32)

# Initialize the similarity scores dictionary
similarity_scores = {}

# Iterate through all the query images in the directory
for i in range(1, 41):
    # Generate the file name for the query image
    query_file = 'i{:02d}.ppm'.format(i)
    query_path = os.path.join(image_dir, query_file)

    # Initialize the similarity scores list for the query image
    similarity_scores[query_file] = []

    # Check if the file exists
    if os.path.exists(query_path):
        # Load the query image
        query_image = cv.imread(query_path)

        # Convert the query image to grayscale
        query_image = (query_image[:,:,0] + query_image[:,:,1] + query_image[:,:,2]) / 3.0

        # Calculate the 3D color histogram for the query image
        query_hist = cv.calcHist([query_image], [0, 1, 2], None, [bins, bins, bins], [0, 2 ** bits, 0, 2 ** bits, 0, 2 ** bits])

        # Normalize the query histogram
        # query_hist = cv.normalize(query_hist, None)
        query_hist = cv.normalize(query_hist, query_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        # Iterate through all the target images in the directory
        for j in range(1, 41):
            # Skip the query image itself
            if i == j:
                continue

            # Generate the file name for the target image
            target_file = 'i{:02d}.ppm'.format(j)
            target_path = os.path.join(image_dir, target_file)

            # Check if the file exists
            if os.path.exists(target_path):
                # Load the target image
                target_image = cv.imread(target_path)

                # Convert the target image to grayscale
                target_image = (target_image[:,:,0] + target_image[:,:,1] + target_image[:,:,2]) / 3.0

                # Calculate the 3D color histogram for the target image
                target_hist = cv.calcHist([target_image], [0, 1, 2], None, [bins, bins, bins], [0, 2 ** bits, 0, 2 ** bits, 0, 2 ** bits])

                # Normalize the target histogram
                # target_hist = cv.normalize(target_hist, None)
                target_hist =  cv.normalize(target_hist, target_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

                # Compute the normalized L1 distance between the query and target histograms
                # distance = cv.norm(query_hist, target_hist, cv.NORM_L1)
                distance = np.sum(np.abs(query_hist - target_hist)) / (2 * 60 * 89)

                # Append the similarity score to the list for the query image
                similarity_scores[query_file].append([distance, target_file])

        # Sort the similarity scores for the query image by distance in ascending order
        similarity_scores[query_file].sort(key = lambda x : x[0])
        # print(similarity_scores[query_file])

        # Select the top 3 similar images based on distance and add up their scores
        similar_images = []
        total_score = 0

        for k in range(3):
            img_num = similarity_scores[query_file][k][1][1:3]
            total_score += crowd_data[i - 1][int(img_num) - 1]
            # Compute the score for the target image based on the crowdsource data
            similar_images.append(similarity_scores[query_file][k][1])

        # print to console
        print('Query image:', query_file)
        print('\tTotal score:', total_score)
        print('\tTop 3 similar images:', ', '.join(similar_images))