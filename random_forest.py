import math
import numpy
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

    def __init__(self, xtrain, ytrain, entropy=None, T=0.01, X=20, feat_subset=20):
        if entropy is None:
            spam = float(numpy.sum(ytrain)) / len(ytrain)
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

        if len(xtrain) <= X or numpy.sum(ytrain) == len(ytrain) or numpy.sum(ytrain) == 0:
            # print 'Number of points in node is %s. Terminating.' % len(xtrain)
            self.is_spam = self.get_majority_label()
            return

        entropies = []
        # bar = pbar(xtrain.shape[1])
        # bar.start()
        subset_of_features = numpy.random.permutation(range(xtrain.shape[1]))[:feat_subset]
        for j in subset_of_features:
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
        if not entropies:
            self.is_spam = self.get_majority_label()
            return
        self.feat, self.val, l_e, r_e, n_e = min(entropies, key=itemgetter(4))

        # print 'New entropy is %0.4f.' % n_e

        if (self.entropy - n_e) < T:
            # print 'Change in entropy is %0.4f. Terminating.' % (self.entropy - n_e)
            self.is_spam = self.get_majority_label()
            return
        lxtrain, lytrain, rxtrain, rytrain = self.splitData()
        self.left_child = DecisionTree(lxtrain, lytrain, l_e, T, X, feat_subset)
        self.right_child = DecisionTree(rxtrain, rytrain, r_e, T, X, feat_subset)

    def error_free_log(self, num):
        err = numpy.seterr(divide='ignore', invalid='ignore')
        lg = numpy.nan_to_num(numpy.log2(num))
        numpy.seterr(**err)
        return lg

    def get_majority_label(self):
        ones = len(self.ytrain[self.ytrain == 1])
        return float(ones)/len(self.ytrain)

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


def shuffle(xtrain, ytrain):
    '''
    Shuffles xtrain and ytrain and returns the tuple
    '''
    shuffle_indices = numpy.random.permutation(range(3065))
    shuffled_xtrain = xtrain[shuffle_indices]
    shuffled_ytrain = ytrain[shuffle_indices]
    return shuffled_xtrain, shuffled_ytrain


def main():
    data = scipy.io.loadmat('spamData.mat')
    xtrain = data['Xtrain']
    ytrain = data['ytrain']
    xtest = data['Xtest']
    ytest = data['ytest']
    trees = []

    ntrees = [25, 50, 100]
    samples = [500, 1000, 2000]
    feats = [15, 30, 45]
    Ts = [0.01, 0.001]
    Xs = [5, 10, 25]
    results = ''
    combos = [x for x in itertools.product(ntrees, samples, feats, Ts, Xs)]
    bar = pbar(len(combos))
    bar.start()
    count = 0
    for num_trees, sample_subset, feat_subset, T, X in combos:
        for i in xrange(num_trees):
            shuffled_xtrain, shuffled_ytrain = shuffle(xtrain, ytrain)
            trees.append(DecisionTree(shuffled_xtrain[:sample_subset], shuffled_ytrain[
                         :sample_subset], None, T, X, feat_subset))
        error = 0
        for i in xrange(len(xtest)):
            sample = xtest[i]
            predictions = [tree.classify(sample) for tree in trees]
            avg_prediction = round(float(sum(predictions))/len(predictions))
            if avg_prediction != ytest[i]:
                error += 1
        error = float(error) / len(xtest)
        results += '%0.4f\t%s\t%s\t%s\t\t%s\t\t%s\n' % (error, T, X, num_trees, sample_subset, feat_subset)
        count += 1
        bar.update(count)
    bar.finish()
    print 'error\tT\tX\tnum_trees\tsample_subset\tfeat_subset\n' + results


if __name__ == "__main__":
    main()
