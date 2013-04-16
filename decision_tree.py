import math
import numpy
import random
import scipy
import scipy.io
from operator import itemgetter
import progressbar
from collections import Counter
import itertools


def pbar(size):
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar


class DecisionTree:

    def __init__(self, xtrain, ytrain, entropy=None, T=0.01, X=3):
        if entropy is None:
            spam = float(numpy.sum(ytrain)) / len(ytrain)
            if spam == 0 or spam == 1:
                entropy = 0
            else:
                entropy = spam * math.log(spam)
                entropy += (1 - spam) * math.log(1 - spam)
                entropy = entropy * -1.0
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.entropy = entropy
        self.left_child = None
        self.right_child = None
        self.is_spam = None

        if len(xtrain) <= X or numpy.sum(ytrain) == len(ytrain) or numpy.sum(ytrain) == 0:
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
            sumL, sumR, lenL, lenR = [], [], [], []
            buckets = sorted(Counter(xtrain_j).items())
            so_far = buckets[0][1]
            for i in xrange(1, len(buckets)):
                left = ytrain_j[:so_far]
                right = ytrain_j[so_far:]
                sumL.append(numpy.sum(left))
                lenL.append(float(len(left)))
                sumR.append(numpy.sum(right))
                lenR.append(float(len(right)))
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
        self.feat, self.val, l_e, r_e, n_e = min(entropies, key=itemgetter(4))

        # print 'New entropy is %0.4f.' % n_e

        if (self.entropy - n_e) < T:
            # print 'Change in entropy is %0.4f. Terminating.' % (self.entropy - n_e)
            self.is_spam = self.get_majority_label()
            return
        lxtrain, lytrain, rxtrain, rytrain = self.splitData()
        if len(lxtrain) == 0 or len(rxtrain) == 0:
            import pdb
            pdb.set_trace()
        self.left_child = DecisionTree(lxtrain, lytrain, l_e, T, X)
        self.right_child = DecisionTree(rxtrain, rytrain, r_e, T, X)

    def error_free_log(self, num):
        err = numpy.seterr(divide='ignore', invalid='ignore')
        lg = numpy.nan_to_num(numpy.log(num))
        numpy.seterr(**err)
        return lg

    def get_majority_label(self):
        ones = len(self.ytrain[self.ytrain == 1])
        zeros = len(self.ytrain[self.ytrain == 0])
        if ones > zeros:
            return True
        elif zeros > ones:
            return False
        else:
            return random.choice((True, False))

    def split(self, feat, val):
        return self.xtrain[:, feat] < val, self.xtrain[:, feat] >= val

    def splitData(self):
        left_indices, right_indices = self.split(self.feat, self.val)
        return (self.xtrain[numpy.array(left_indices)],
                self.ytrain[numpy.array(left_indices)],
                self.xtrain[numpy.array(right_indices)],
                self.ytrain[numpy.array(right_indices)])

    def child_entropy(self, indices):
        labels = self.ytrain[indices]
        if len(labels) == 0:
            return 0
        spam = float(sum(labels)) / len(labels)
        if spam == 0 or spam == 1:
            entropy = 0
        else:
            entropy = spam * math.log(spam)
            entropy += (1 - spam) * math.log(1 - spam)
            entropy = entropy * -1.0
        return entropy

    def get_entropy(self, feat, val):
        '''
        Calculates entropy: H(x) = Summation( P(x) * log(P(x)) ) for all x
        '''
        left_indices, right_indices = self.split(feat, val)
        left_entropy = self.child_entropy(left_indices)
        right_entropy = self.child_entropy(right_indices)
        entropy = left_entropy * len(left_indices) / len(self.xtrain)
        entropy += right_entropy * len(right_indices) / len(self.xtrain)
        return left_entropy, right_entropy, entropy

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


def main():
    data = scipy.io.loadmat('spamData.mat')
    xtrain = data['Xtrain']
    ytrain = data['ytrain']
    xtest = data['Xtest']
    ytest = data['ytest']
    t_values = [0.1, 0.01, 0.001, 0.0001]
    x_values = [1, 3, 5, 10, 25, 50, 100]
    print 'T\tX\tError'
    for t, x in itertools.product(t_values, x_values):
        tree = DecisionTree(xtrain, ytrain, None, t, x)
        error = 0
        for i in xrange(len(xtest)):
            sample = xtest[i]
            if tree.classify(sample) != ytest[i]:
                error += 1
        print '%s\t%s\t%0.4f' % (t, x, float(error) / len(xtest))


if __name__ == "__main__":
    main()
