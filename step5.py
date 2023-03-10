import cv2 as cv
import numpy as np
import os

from step1 import similarity_scores
step1_scores = similarity_scores
from step2 import similarity_scores
step2_scores = similarity_scores
from step3 import similarity_scores
step3_scores = similarity_scores
from step4 import similarity_scores
step4_scores = similarity_scores

# Step 5: Overall distance

# Set the directory where the images are located
image_dir = './data/images/'

# Load the crowdsource data
crowd_data = np.loadtxt('./data/Crowd.txt', dtype=np.int32)

# Initialize the similarity scores dictionary
similarity_scores = {}

# Initialize the total score, which is the total crowd count
total_score = 0

C_weight = 0.6
T_weight = 0.1
S_weight = 0.2
Y_weight = 0.1

def minimize_distance(C_score, T_score, S_score, Y_score, learning_rate=0.005, max_iterations=70, tolerance=0.8):
    # initialize weights
    C_weight = T_weight = S_weight = Y_weight = 0.25

    # repeat until convergence or maximum iterations reached
    for i in range(max_iterations):
        # calculate distance using current weights
        distance = C_weight * C_score + T_weight * T_score + S_weight * S_score + Y_weight * Y_score

        # calculate gradients
        C_gradient = C_score
        T_gradient = T_score
        S_gradient = S_score
        Y_gradient = Y_score

        # update weights using gradient descent
        C_weight = max(min(C_weight - learning_rate * C_gradient, 1), 0)
        T_weight = max(min(T_weight - learning_rate * T_gradient, 1), 0)
        S_weight = max(min(S_weight - learning_rate * S_gradient, 1), 0)
        Y_weight = max(min(Y_weight - learning_rate * Y_gradient, 1), 0)

        # normalize weights
        total_weight = C_weight + T_weight + S_weight + Y_weight
        C_weight /= total_weight
        T_weight /= total_weight
        S_weight /= total_weight
        Y_weight /= total_weight

        # check for convergence
        if abs(distance - (C_weight * C_score + T_weight * T_score + S_weight * S_score + Y_weight * Y_score)) < tolerance:
            break

    # return optimal weights
    return C_weight, T_weight, S_weight, Y_weight

# Create and write in the HTML file
with open("step5_results.html", "w") as file:
    file.write("<html>\n")
    file.write("<head>\n")
    file.write("<title>Step 5: Shape Similarity Results</title>\n")
    file.write("<h1>Step 5: Shape Similarity Results</h1>\n")
    file.write("<p><strong>NOTE</strong>: The total score, accuracy score, and happiness score "
               "are displayed at the very end of the file.</p>\n")
    file.write("</head>\n")
    file.write("<body>\n")

# Iterate through all the query images in the directory
for i in range(1, 41):
    # Generate the file name for the query image
    query_file = 'i{:02d}.ppm'.format(i)
    query_path = os.path.join(image_dir, query_file)

    step1_scores[query_file].sort(key=lambda x: x[1])
    step2_scores[query_file].sort(key=lambda x: x[1])
    step3_scores[query_file].sort(key=lambda x: x[1])
    step4_scores[query_file].sort(key=lambda x: x[1])

    # Initialize the similarity scores list for the query image
    similarity_scores[query_file] = []

    # Check if the file exists
    if os.path.exists(query_path):
        # Load the query image
        query_image = cv.imread(query_path)

        index = -1

        # Iterate through all the target images in the directory
        for j in range(1, 41):
            # Skip the query image itself
            if i == j:
                continue

            index += 1

            # Generate the file name for the target image
            target_file = 'i{:02d}.ppm'.format(j)
            target_path = os.path.join(image_dir, target_file)

            # Check if the file exists
            if os.path.exists(target_path):
                # Load the target image
                target_image = cv.imread(target_path)

                if index < 40:
                    C_score = step1_scores[query_file][index][0]
                    T_score = step2_scores[query_file][index][0]
                    S_score = step3_scores[query_file][index][0]
                    Y_score = step4_scores[query_file][index][0]

                # C_weight, T_weight, S_weight, Y_weight = minimize_distance(C_score, T_score, S_score, Y_score)

                # Compute the overall normalized L1 distance
                distance = C_weight * C_score + T_weight * T_score + S_weight * S_score + Y_weight * Y_score

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

        # Add total score for a row to the grand total
        total_score += score

        # # print to console
        # print('Query image:', query_file)
        # print('\tScore:', score)
        # print('\tTop 3 similar images:', ', '.join(similar_images))

        with open("step5_results.html", "a") as file:
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
print('Step 5 Accuracy:', accuracy)
print('Step 5 Total score:', total_score)
print('A: {}, B: {}, C: {}, D: {}'.format(C_weight, T_weight, S_weight, Y_weight))
print(C_weight + T_weight + S_weight + Y_weight)

# close the HTML file
with open("step5_results.html", "a") as file:
    file.write("<h3>Total score: {}</h3>\n".format(total_score))
    file.write("<h3>Accuracy: {}%</h3>\n".format(accuracy))
    file.write("<h3>Happiness: 39\n")
    file.write("</body>\n")
    file.write("</html>\n")
