'''
PHY407 FINAL PROJECT
HOPFIELD NETWORKS by Lyndon Boone

Code includes functions for the operation of the Hopfield network, as well as
code for two experiments done for the project. Running this file will train the
network on the template images in templates.npy and run the simulations
corresponding to the experiments, generating figures that go along with them.

The entire program takes about 10 mins to run on my computer (Intel Core i5).
The "training" of the network takes almost negligible time (the network is
quite small and the learning rule is simple), but what takes a long time is
running the experiments with a large number of trials.

***If you want the program to run in a shorter amount of time, modify the
parameter N from 18444 to some smaller number (although the error bars in
the plots will no longer be valid).

***Make sure to include the templates.npy file in the same directory as this
fild when you run it.
'''

import numpy as np
import matplotlib.pyplot as plt

# load templates from npy file
templates = np.load('templates.npy')
templateNames = ['A', 'B', 'C', 'H', 'T']

# flatten template images to make training dataset
trainData = np.reshape(templates, (5, 100))


def threshold(a):
    '''
    Threshold activation function used to determine whether the sum of inputs
    into a network node are sufficient to activate the node (state +1). The
    threshold is zero such that if the sum of inputs is greater than 0, the
    node activation gets +1, else if the sum of inputs is less than 0, the
    node activation gets -1.

    Input:
        a:          sum of inputs to the node

    Returns:        activation of the binary neuron (+1 or -1)
    '''
    return 1 if a >= 0 else -1


def updateState(v, W, index):
    '''
    Updates the state of a single neuron by computing its activation (sum of
    inputs) and passing that value through a threshold activation function.

    Inputs:
        v:          flattened array of neuron states
        W:          weight matrix (2D array)
        index:      index of the neuron to update

    Returns:
        newState:   flattened array with updated neuron state
    '''
    activation = np.matmul(v, W[:, index])    # compute the activation
    newState = threshold(activation)         # threshold activation function
    return newState


def asyncUpdate(v, W):
    '''
    Update each neuron in the network once, asynchronously. Return a new array
    consisting of the updated neuron states after each node has been updated.

    Inputs:
        v:          flattened array of neuron states
        W:          weight matrix (2D array)

    Returns:
        vNew:       new flattened array of neuron states with each node updated
                    asynchronously
    '''
    neurons = len(v)      # number of neurons in flattened array
    vNew = np.copy(v)     # create copy of original state vector

    # shuffle the order of neurons to update
    updateOrder = np.random.permutation(np.arange(neurons))

    # iterate over permuted list of indices, update each neuron individually
    for i in range(neurons):
        index = updateOrder[i]
        vNew[index] = updateState(vNew, W, index)
    return vNew


def findMinState(im, W, maxUpdates=100, return_iterations=False,
                 return_success=False):
    '''
    Update the state of the Hopfield network until convergence, or the maximum
    number of updates is reached.

    Inputs:
        im:                    input image (2D numpy array, not flattened)
        W:                     trained weight matrix (2D array)
        maxUpdates:            maximum number of updates to try and reach
                               convergence (optional, default=100)
        return_iterations:     boolean, whether to return the number of updates
                               taken before successful convergence

    Returns:
        imOut:                 converged network state (un-flattened 2D array)
        updates:               number of updates taken before convergence
        success:               1 if the system converged, else -1
    '''
    # dimensions of input image
    N = im.shape[0]
    M = im.shape[1]

    # flatten input image to pass into network
    imFlat = np.ndarray.flatten(im)
    success = -1                          # indicates converge before maxUpdate
    for updates in range(maxUpdates):
        imTemp = asyncUpdate(imFlat, W)   # full round of asynchronous updates
        truth = imTemp == imFlat          # compare new array to previous
        if truth.all():
            success = 1
            break                         # if no change in state, break loop
        else:
            imFlat = imTemp               # else, update the state
    imOut = np.reshape(imFlat, (N, M))    # un-flatten image before return
    if return_iterations and return_success:
        return imOut, updates + 1, success
    elif return_iterations:
        return imOut, updates + 1
    elif return_success:
        return imOut, success
    else:
        return imOut


def energy(im, W):
    '''
    Compute the energy of the system. (Not used)

    Inputs:
        im:                    input image (2D numpy array, not flattened)
        W:                     trained weight matrix (2D array)

    Returns:
        E:                     energy of the system
    '''
    imFlat = np.ndarray.flatten(im)               # flatten input image
    E = -np.sum(W*imFlat*np.transpose(imFlat))    # compute energy
    return E


def learnRepr(vTrain):
    '''
    Learn the weights of the system to make the input data local minima of the
    energy function using the Hebb-inspired learning rule.

    Inputs:
        vTrain:      "training" data consisting of retrieval states desired to
                     be minima of the energy function

    Returns:
        W:           trained weight matrix (2D array)
    '''
    N = vTrain.shape[0]      # number of retrieval states
    D = len(vTrain[0])       # size of each retrieval state
    W = np.zeros((D, D))     # initialize weight matrix
    for i in range(D):
        # compute upper triangular elements (elements along diagonal stay 0)
        for j in range(i+1, D):
            for n in range(N):
                W[i,j] += (1/N)*vTrain[n][i]*vTrain[n][j]   # learning rule
            W[j,i] = W[i,j]  # symmetric weights
    return W


def addNoise(im, p=0.1):
    '''
    Generate a corrupted version of the input image.

    Inputs:
        im:     Input image (2D array)
        p:      Noise level (probability of flipping each pixel) (default 0.1)

    Returns:
        imOut:  Corrupted image (2D array)
    '''
    imOut = np.copy(im)                # copy original image
    # loop over each pixel
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            rand = np.random.rand()    # generate random number -> [0,1)
            if rand < p:
                imOut[i,j] *= -1       # flip pixel with probability p
    return imOut


# begin simulations

W = learnRepr(trainData)    # learn weight matrix using train data

N = 18444                   # trials to generate 95% CI (see report appendix)
noiseLevels = np.array([0.1,0.2,0.3,0.4,0.5])   # corruption levels to test

# array to store recall accuracies across different letters and noise levels
results = np.zeros((len(templates), len(noiseLevels)))

# begin main simulation
# iterate over noise levels (corruption parameter p)
for i in range(len(noiseLevels)):
    # iterate over template letter images
    for j in range(len(templates)):
        count = 0    # initialize count of correct recall associations
        # iterate over number of trials
        for k in range(N):
            # generate corrupted test image
            imTest = addNoise(templates[j], p=noiseLevels[i])
            # let network converge to stable state
            imUpdate = findMinState(imTest, W)
            # compare converged state to original image test was generated from
            truth = imUpdate == templates[j]
            if truth.all():
                count += 1         # if the same as original image, add 1
        # calculate recall accuracy for letter and noise level
        percentCorrect = count/N
        results[j,i] = percentCorrect    # store in array
        print('Recall accuracy on letter {} at {:.0f}% noise level: {:.2f}%'.format(templateNames[j], 100*noiseLevels[i], 100*percentCorrect))
    print('Average recall accuracy across all letters at {:.0f}% noise level: {:.2f}%\n'.format(100*noiseLevels[i], 100*np.mean(results, axis=0)[i]))

# generate plot of recall accuracy vs. image corruption
plt.figure()
for i in range(len(templates)):
    plt.errorbar(100*noiseLevels, 100*results[i], yerr=1,
                 label='{}'.format(templateNames[i]))
plt.plot(100*noiseLevels, 100*np.mean(results, axis=0), 'k--', label='Avg.')
plt.xlabel('Image corruption (%)')
plt.ylabel('Recall accuracy (%)')
plt.title('Recall accuracy as a function of image corruption')
plt.grid()
plt.legend()
plt.show()

# run another simulation on completely random input images (noise)

randomRecalls = np.zeros(2*len(templates) + 1)  # initialize storage array
# loop over number of trials
for i in range(N):
    imTest = np.random.randint(0, 2, (10,10))   # generate random test image
    imTest[imTest == 0] = -1
    imUpdate = findMinState(imTest, W)          # let network converge
    # test if converged state is the same as any template images
    for j in range(len(templates)):
        truth = imUpdate == templates[j]
        if truth.all():
            randomRecalls[j] += 1               # store values in array
            break
        # test to see if system converged to inverse of template image
        elif np.logical_not(truth).all():
            randomRecalls[j+5] += 1             # store values in array
            break
randomRecalls[-1] = N - np.sum(randomRecalls)   # 'other' final state
randomRecalls = randomRecalls / N               # go from count to percentage

# count number of '1' bits in each template image
templatesCopy = np.copy(templates)
templatesCopy[templatesCopy == -1] = 0
numOnes = np.sum(templatesCopy, axis=(1,2))

labels = ['A', 'B', 'C', 'H', 'T', 'Other']   # labels for bar chart

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

# plot bar chart with frequency of converged state
fig, ax = plt.subplots()
rects1 = ax.bar(x[0:5] - width/2, 100*randomRecalls[0:5], width,
                label='template image', color='red')
rects2 = ax.bar(x[0:5] + width/2, 100*randomRecalls[5:-1], width,
                label='inverse of template', color='blue')
rects3 = ax.bar(x[5:], 100*randomRecalls[-1:], width, color='gray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Frequency (%)')
ax.set_title('Final convergence state for randomly generated images')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid()

# generate a second y axis quantifying number of '1' bits in each template
ax2 = ax.twinx()
color = 'green'
ax2.set_ylabel('Percent  binary \'1\' (%)', color=color, rotation=270)
ax2.plot(x[0:5], numOnes, color=color)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelcolor=color)

# generate figure of all templates side by side
plt.figure()
for i in range(len(templates)):
    plt.subplot(151 + i)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.imshow(templates[i])
plt.tight_layout()
plt.show()

# generate figures with examples of good convergence for each noise level
for i in range(len(noiseLevels)):
    noiseImages = []
    for j in range(len(templates)):
        for k in range(300):
            imTest = addNoise(templates[j], p=noiseLevels[i])
            imUpdate = findMinState(imTest, W)
            truth = imUpdate == templates[j]
            if truth.all():
                noiseImages.append(imTest)
                break
            elif k == 299:
                print('Unable to recall after 300 attempts.')
    plt.figure()
    for i in range(len(templates)):
        plt.subplot(2, 5, 1 + i)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.imshow(noiseImages[i])
        plt.subplot(2, 5, 6 + i)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.imshow(templates[i])
    plt.tight_layout()
    plt.show()

# generate some figures with spurious states
testImages = []
convergedImages = []
count = 0
while count != 5:
    foundMatch = 0
    imTest = np.random.randint(0, 2, (10,10))
    imUpdate, success = findMinState(imTest, W, return_success=True)
    if success == -1:
        continue
    for j in range(len(templates)):
        truth = imUpdate == templates[j]
        if truth.all():
            foundMatch = 1
            break
        # test to see if system converged to inverse of template image
        elif np.logical_not(truth).all():
            foundMatch = 1
            break
    if not foundMatch:
        count += 1
        testImages.append(imTest)
        convergedImages.append(imUpdate)
plt.figure()
for i in range(len(templates)):
    plt.subplot(2, 5, 1 + i)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.imshow(testImages[i])
    plt.subplot(2, 5, 6 + i)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.imshow(convergedImages[i])
plt.tight_layout()
plt.show()
