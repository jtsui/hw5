import math
import numpy
import random
import scipy
import scipy.io
from operator import itemgetter
import progressbar
from collections import Counter


def pbar(size):
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar


class DecisionTree:

    def __init__(self, xtrain, ytrain, weights, entropy=None, T=0.01, X=3):
        if entropy is None:
            spam = float(numpy.sum(ytrain.clip(0))) / len(ytrain)
            if spam == 0 or spam == 1:
                entropy = 0
            else:
                entropy = spam * math.log(spam, 2)
                entropy += (1 - spam) * math.log((1 - spam), 2)
                entropy = entropy * -1.0
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.entropy = entropy
        self.left_child = None
        self.right_child = None
        self.is_spam = None
        self.weights = weights

        if len(xtrain) <= X or numpy.sum(ytrain) == len(ytrain) or numpy.sum(ytrain) == -len(ytrain):
            # print 'Number of points in node is %s. Terminating.' % len(xtrain)
            self.is_spam = self.get_majority_label()
            return

        entropies = []
        # bar = pbar(xtrain.shape[1])
        # bar.start()
        for j in xrange(xtrain.shape[1]):
            xtrain_j_idx = xtrain[:, j].argsort()
            ytrain_j = ytrain[xtrain_j_idx]
            xtrain_j = xtrain[:, j][xtrain_j_idx]
            weights_j = weights[xtrain_j_idx]
            sumL, sumR, lenL, lenR = [], [], [], []
            buckets = sorted(Counter(xtrain_j).items())
            so_far = buckets[0][1]
            for i in xrange(1, len(buckets)):
                left = ytrain_j[:so_far]
                right = ytrain_j[so_far:]
                left_weight = weights_j[:so_far]
                right_weight = weights_j[so_far:]
                sumL.append(numpy.sum(left_weight[left == 1]))
                lenL.append(float(numpy.sum(left_weight)))
                sumR.append(numpy.sum(right_weight[right == 1]))
                lenR.append(float(numpy.sum(right_weight)))
                so_far += buckets[i][1]

            sumL = numpy.array(sumL)
            sumR = numpy.array(sumR)
            lenL = numpy.array(lenL)
            lenR = numpy.array(lenR)
            probL = sumL / lenL
            probR = sumR / lenR
            left_entropies = -(probL * self.error_free_log(probL) + (1-probL) * self.error_free_log(1-probL))
            right_entropies = -(probR * self.error_free_log(probR) + (1-probR) * self.error_free_log(1-probR))
            entropies_j = (lenL/(lenL+lenR))*left_entropies+(lenR/(lenL+lenR))*right_entropies
            entropies += zip([j] * (len(buckets) - 1), [x for x, y in buckets][1:], list(
                left_entropies), list(right_entropies), list(entropies_j))
            # bar.update(j)
        # bar.finish()
        if not entropies:
            self.is_spam = self.get_majority_label()
            return
        self.feat, self.val, l_e, r_e, n_e = min(entropies, key=itemgetter(4))

        # print 'New entropy is %0.4f.' % n_e
        
        if (self.entropy - n_e) < T:
            #print 'Change in entropy is %0.4f. Terminating.' % (self.entropy - n_e)
            self.is_spam = self.get_majority_label()
            return
        lxtrain, lytrain, rxtrain, rytrain = self.splitData()
        self.left_child = DecisionTree(lxtrain, lytrain, weights, l_e, T, X)
        self.right_child = DecisionTree(rxtrain, rytrain, weights, r_e, T, X)

    def error_free_log(self, num):
        err = numpy.seterr(divide='ignore', invalid='ignore')
        lg = numpy.nan_to_num(numpy.log2(num))
        numpy.seterr(**err)
        return lg

    def get_majority_label(self):
        ones = numpy.sum(self.weights[self.ytrain == 1])
        zeros = numpy.sum(self.weights[self.ytrain == -1])
        if ones > zeros:
            return 1
        elif zeros > ones:
            return -1
        else:
            return random.choice((-1, 1))

    def split(self, feat, val):
        return self.xtrain[:, feat] < val, self.xtrain[:, feat] >= val

    def splitData(self):
        left_indices, right_indices = self.split(self.feat, self.val)
        return (self.xtrain[numpy.array(left_indices)],
                self.ytrain[numpy.array(left_indices)],
                self.xtrain[numpy.array(right_indices)],
                self.ytrain[numpy.array(right_indices)])

    def classify(self, sample):
        '''
        evaluate the sample, returning True for spam and False for ham
        '''
        if self.left_child is not None and self.right_child is not None:
            sample_val = sample[self.feat]
            if sample_val < self.val:
                return self.left_child.classify(sample)
            else:
                return self.right_child.classify(sample)
        else:
            return self.is_spam


def update_weights(current_weights, wrong_indices, alpha):
    right_factor = math.exp(-alpha)
    wrong_factor = math.exp(alpha)
    new_weights = []
    for i in xrange(len(current_weights)):
        factor = wrong_factor if i in wrong_indices else right_factor
        new_weights.append(current_weights[i] * factor)
    new_weights = numpy.array(new_weights)
    return new_weights/numpy.sum(new_weights)


def main():
    data = scipy.io.loadmat('spamData.mat')
    xtrain = data['Xtrain']
    ytrain = data['ytrain']
    ytrain = ytrain.astype(int)
    ytrain = numpy.where(ytrain == 1, ytrain, -1)
    xtest = data['Xtest']
    ytest = data['ytest']
    ytest = ytest.astype(int)
    ytest = numpy.where(ytest == 1, ytest, -1)
    iterations = 7
    t_val = 0.2
    x_val = 200
    tree_weights = []
    trees = []
    weights = [1.0/len(ytrain)] * len(ytrain)
    weights = numpy.array([weights]).transpose()
    bar = pbar(iterations)
    bar.start()
    for t in range(iterations):
        tree = DecisionTree(xtrain, ytrain, weights, None, t_val, x_val)
        wrong_indices = []
        error = 0.0
        for i in xrange(len(xtrain)):
            if tree.classify(xtrain[i]) != ytrain[i]:
                wrong_indices.append(i)
                error += weights[i]
        #error = float(error) / len(xtrain)
        print 'Tree error %0.4f' % error
        trees.append(tree)
        alpha = 0.5 * math.log((1 - error)/error)
        weights = update_weights(weights, wrong_indices, alpha)
        tree_weights.append(alpha)
        bar.update(t)
    bar.finish()

    error = 0
    for i in xrange(len(xtest)):
        sample = xtest[i]
        predictions = [t.classify(sample) * w for t, w in zip(trees, tree_weights)]
        prediction = 1 if sum(predictions) > 0 else -1
        if prediction != ytest[i]:
            error += 1
    error = float(error) / len(xtest)
    print 'T\tX\tError'
    print '%s\t%s\t%0.4f' % (t_val, x_val, error)


if __name__ == "__main__":
    main()
