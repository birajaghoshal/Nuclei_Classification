import time
from active_learner import ActiveLearner


class BootStrap_Learner(ActiveLearner):
    def run(self):
        """ Runs the active learner with random data updates.
        :return: The lists of testing metrics from each iteration.
        """

        # Defines the lists to store the metrics.
        accuracies, mean_accuracies, recalls, precisions, f1_scores, losses = [], [], [], [], [], []
        start_time = time.clock()

        # Loops until all the data has been read.
        while len(self.data.data_y) != 0:
            self.log("\nCycle " + str(len(accuracies) + 1))