import cv2 as cv
import numpy as np
import os

# Step 1: color distance (C)

# Set the directory where the images are located
image_dir = './data/images/'

# Set the number of bits for each channel
bits = 3  # the "Goldilocks"

# Set the number of bins for each channel
bins = 2 ** bits

# Load the crowdsource data
crowd_data = np.loadtxt('./data/Crowd.txt', dtype=np.int32)

# Initialize the similarity scores dictionary
similarity_scores = {}

# Initialize the total_score, which is the total crowd count
total_score = 0

# Create and write in the HTML file
with open("step1_results.html", "w") as file:
    file.write("<html>\n")
    file.write("<head>\n")
    file.write("<title>Step 1: Color Similarity Results</title>\n")
    file.write("<h1>Step 1: Color Similarity Results</h1>\n")
    file.write("<p><strong>NOTE</strong>: The total score, accuracy score, and happiness score "
               "are displayed at the very end of the file.</p>\n")
    file.write("</head>\n")
    file.write("<body>\n")

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

        # Calculate the 3D color histogram for the query image
        query_hist = cv.calcHist([query_image], [0, 1, 2], None, [bins, bins, bins], [0, 255, 0, 255, 0, 255])

        # Normalize the query histogram
        # query_hist = cv.normalize(query_hist, query_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

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

                # Calculate the 3D color histogram for the target image
                target_hist = cv.calcHist([target_image], [0, 1, 2], None, [bins, bins, bins], [0, 255, 0, 255, 0, 255])

                # Compute the normalized L1 distance between the query and target histograms
                distance = np.sum(np.abs(query_hist - target_hist)) / (2 * 60 * 89)

                # Append the similarity score to the list for the query image
                similarity_scores[query_file].append([distance, target_file])

        # Sort the similarity scores for the query image by distance in ascending order
        similarity_scores[query_file].sort(key = lambda x : x[0])
        # print(similarity_scores[query_file])

        # Select the top 3 similar images based on distance and add up their scores
        similar_images = []
        score = 0

        for k in range(3):
            img_num = similarity_scores[query_file][k][1][1:3]
            score += crowd_data[i - 1][int(img_num) - 1]
            # Compute the score for the target image based on the crowdsource data
            similar_images.append(similarity_scores[query_file][k][1])

        # Add score for a row to the total score
        total_score += score

        # # print to console
        # print('Query image:', query_file)
        # print('\tScore:', score)
        # print('\tTop 3 similar images:', ', '.join(similar_images))

        with open("step1_results.html", "a") as file:
            file.write("<hr><h3>Query image: {}</h3>\n".format(query_file[1:3]))
            file.write("<p>Score: {}</p>\n".format(score))
            file.write("<div>\n")
            query_path = os.path.join(image_dir, 'i{:02d}.jpg'.format(i))
            file.write("<div style='display:flex;'>\n")
            file.write("<img src='{}' height='80px' style='margin-right: 60px;'>\n".format(query_path))

            for idx, sim_img in enumerate(similar_images):
                sim_path = os.path.join(image_dir, 'i{}.jpg'.format(sim_img[1:3]))
                crowd_count = crowd_data[i - 1][int(sim_img[1:3]) - 1]
                score = crowd_count / (idx + 1)
                file.write("<div>\n")
                file.write("<img src='{}' height='80px' style='margin-right: 40px;'>\n".format(sim_path))
                file.write("<p>Similar image {}: {}</p>\n".format(idx + 1, sim_img[1:3]))
                file.write("<p>(Crowd count: {})</p>\n".format(crowd_count))
                file.write("</div>\n")

            file.write("</div>\n")

accuracy = total_score / 25200 * 100  # Goal: between 30% - 40%

# print to console
print('Step 1 Accuracy:', accuracy)
print('Step 1 Total score:', total_score)

# close the HTML file
with open("step1_results.html", "a") as file:
    file.write("<h3>Total score: {}</h3>\n".format(total_score))
    file.write("<h3>Accuracy: {}%</h3>\n".format(accuracy))
    file.write("<h3>Happiness: 39</h3>\n")
    file.write("</body>\n")
    file.write("</html>\n")