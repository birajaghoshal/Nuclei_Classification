import math
import time
import numpy as np
from active_learner import ActiveLearner


class UncertaintyLearner(ActiveLearner):
    def run(self):
        """ Runs the active learner with data updates based on uncertainty.
        :return: The list of testing metrics from each iteration.
        """

        # Defines the lists to store the metrics.
        accuracies, mean_accuracies, recalls, precisions, f1_scores, losses = [], [], [], [], [], []
        start_time = time.clock()

        # Loops until all the data has been read.
        while len(self.data.data_y) != 0:
            self.log("\nCycle " + str(len(accuracies) + 1))

            # Trains a model with the training data.
            accuracy, mean_accuracy, recall, precision, f1_score, loss = self.model.train(self.data)

            # Adds the metrics to the lists.
            accuracies.append(accuracy)
            mean_accuracies.append(mean_accuracy)
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1_score)
            losses.append(loss)

            # Makes predictions for each cell and selects the most uncertain cells.
            predictions, labels = self.model.predict(self.data, np.average)
            if self.config.update_size * self.config.cell_patches< len(self.data.data_y):
                update = self.config.update_size
            else:
                update = len(self.data.data_x) // self.config.cell_patches

            uncertainties = []
            for prediction in predictions:
                uncertainty = max(prediction)
                for i in range(self.config.num_classes):
                    uncertainty -= prediction[i] * math.log(prediction[i])
                uncertainties.append(uncertainty)


            indices = [i[1] for i in sorted(((value, index) for index, value in enumerate(uncertainties)),
                                            reverse=True)[:update]]
            self.data.set_training_data(indices)

            predictions = np.delete(predictions, indices, axis=0)
            labels = np.delete(labels, indices)

            if self.config.pseudo_labels and len(self.data.data_y) != 0:
                self.data.add_pesudo_labels(predictions, labels)
            else:
                self.data.pesudo_indices = []

            self.log("\n\n")

        # Trains the model with all the data.
        accuracy, mean_accuracy, recall, precision, f1_score, loss = self.model.train(self.data)

        # Adds the metrics to the lists.
        accuracies.append(accuracy)
        mean_accuracies.append(mean_accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1_score)
        losses.append(loss)

        # Logs the metrics.
        self.log("Accuracies: " + str(accuracies) + "\n")
        self.log("Mean Class Accuracies: " + str(mean_accuracies) + "\n")
        self.log("Recalls: " + str(recalls) + "\n")
        self.log("Precisions: " + str(precisions) + "\n")
        self.log("F1 Scores: " + str(f1_scores) + "\n")
        self.log("Losses: " + str(losses))
        self.log("Cycles: " + str(len(accuracies)) + " Time: " + str(time.clock() - start_time))

        # Returns the list of metrics.
        return accuracies, mean_accuracies, recalls, precisions, f1_scores, losses
