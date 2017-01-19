from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import math
import operator
import graph

def load_data_set(file_name):
    data = np.loadtxt(file_name,dtype=str,delimiter=",")
    return pd.DataFrame(np.array(data))

def split_train_test(data_set, split=0.8):
    msk = np.random.rand(len(data_set)) < split
    training = data_set[msk]
    test = data_set[~msk]
    num_training_set = training[[0, 1, 2, 3, 4, 5, 6, 7, 8]].apply(pd.to_numeric)
    num_test_set = test[[0, 1, 2, 3, 4, 5, 6, 7, 8]].apply(pd.to_numeric)
    return num_training_set, num_test_set

def find_neighbors(training_set, test_element, k):
	distances = []
	length = len(test_element)-1
	for x in range(len(training_set)):
		dist = get_euclidean_distance(test_element, training_set[x], length)
		distances.append((training_set[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def get_euclidean_distance(x1, x2, length):
	distance = 0
	for x in range(length):
		distance += pow((x1[x] - x2[x]), 2)
	return math.sqrt(distance)

def get_response(neighbors):
	class_vote = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in class_vote:
			class_vote[response] += 1
		else:
			class_vote[response] = 1
	sorted_votes = sorted(class_vote.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sorted_votes[0][0]

def calculate_accuracy(test_set, predictions):
	correct = 0
	for x in range(len(test_set)):
		if test_set[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(test_set))) * 100.0

def main():
    # Load Sample Data & Training - Test data preparation
    data_set = load_data_set('iris.data')

    split = 0.5
    training_set, test_set = split_train_test(data_set, split)


    print 'Train set: ' + repr(len(training_set))
    print 'Test set: ' + repr(len(test_set))

    # Select Sepal Length & width (2,4)
    X = data_set[[2,4]].apply(pd.to_numeric)

    # Select classified label
    Y = data_set[8].apply(pd.to_numeric)

    # Data visualization
    graph.draw_2d_plot(1, 8, 6, 'Sepal length', 'Sepal width', np.array(Y), np.array(X), np.array(Y))

    num_data_set = data_set[[0, 1, 2, 3, 4, 5, 6, 7]].apply(pd.to_numeric)

    X_reduced = PCA(n_components=3).fit_transform(np.array(num_data_set))
    graph.draw_3d_plot(2, "First three PCA directions of Data", 8, 6, "1st eigenvector", "2nd eigenvector", "3rd eigenvector", np.array(Y), X_reduced, np.array(Y))

    predictions = []
    test_set_array = np.array(test_set)
    train_set_array = np.array(training_set)

    results_X = []
    results_Y = []
    k = 10
    for x in range(len(test_set_array)):
        neighbors = find_neighbors(train_set_array, test_set_array[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test_set_array[x][-1]))
    accuracy = calculate_accuracy(test_set_array, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    results_X.extend(test_set_array[:,0:8])
    results_X.extend(train_set_array[:,0:8])

    results_Y.extend(predictions)
    results_Y.extend(train_set_array[:,8])

    X_reduced = PCA(n_components=3).fit_transform(np.array(results_X))
    graph.draw_3d_plot(3, "First three PCA directions - with Predicted Results", 8, 6, "1st eigenvector", "2nd eigenvector", "3rd eigenvector", np.array(results_Y), X_reduced, np.array(results_Y))

    graph.draw_pyplot()
    # End of Data Visualization
main()

