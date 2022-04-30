import abc
from typing import Tuple
import pandas as pd
import numpy as np
import sklearn
import datetime
from scipy import sparse
import scipy.sparse.linalg as linalg
import math


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

    def rmse(self, true_ratings: pd.DataFrame) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        sum = 0
        for row in true_ratings.index:
            a = true_ratings['rating'][row]
            b = self.predict(true_ratings['user'][row], true_ratings['item'][row], true_ratings['timestamp'][row])
            sum += (a - b) ** 2
        return (sum / len(true_ratings.index)) ** 0.5


class BaselineRecommender(Recommender):
    def __init__(self, ratings: pd.DataFrame):
        self.train = None
        self.users_avg_rating = None
        self.items_avg_rating = None
        self.R_hat = None
        super().__init__(ratings)

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.train = ratings.copy()
        self.R_hat = np.mean(self.train['rating'].values)
        self.train['rating'] = self.train['rating'] - self.R_hat
        self.users_avg_rating = self.train.pivot(index='user', columns='item', values='rating')
        self.users_avg_rating['mean'] = self.users_avg_rating.mean(axis=1)
        self.items_avg_rating = self.train.pivot(index='item', columns='user', values='rating')
        self.items_avg_rating['mean'] = self.items_avg_rating.mean(axis=1)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raw_prediction = self.R_hat + self.users_avg_rating.loc[user]['mean'] + self.items_avg_rating.loc[item]['mean']
        if raw_prediction > 5:
            return 5
        elif raw_prediction < 0.5:
            return 0.5
        else:
            return raw_prediction


class NeighborhoodRecommender(Recommender):
    def __init__(self, ratings: pd.DataFrame):
        self.baseline_recommender = None
        self.similarity_matrix = None
        self.number_of_users = None
        self.number_of_items = None
        super().__init__(ratings)

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.baseline_recommender = BaselineRecommender(ratings)
        self.number_of_users = self.baseline_recommender.users_avg_rating.shape[0]
        self.number_of_items = self.baseline_recommender.items_avg_rating.shape[0]
        self.similarity_matrix = self.cosine_similarity()

    def cosine_similarity(self):
        users_vector_dict = {}
        users_mask_dict = {}
        for record in self.baseline_recommender.train.index:
            user = int(self.baseline_recommender.train['user'][record])
            item = int(self.baseline_recommender.train['item'][record])
            rating = self.baseline_recommender.train['rating'][record]
            if user in users_vector_dict:
                users_vector_dict[user][item] = rating
                users_mask_dict[user][item] = 1
            else:
                users_vector_dict[user] = np.zeros(self.number_of_items)
                users_mask_dict[user] = np.zeros(self.number_of_items)
                users_vector_dict[user][item] = rating
                users_mask_dict[user][item] = 1

        similarity_matrix = np.eye(self.number_of_users)
        for i in range(self.number_of_users):
            for j in range(i):
                mask = users_mask_dict[i] * users_mask_dict[j]
                if np.all((mask == 0)):
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0
                else:
                    d = users_vector_dict[i].dot(users_vector_dict[j])
                    n = np.linalg.norm(mask * users_vector_dict[i]) * np.linalg.norm(mask * users_vector_dict[j])
                    similarity = d / n
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        closest_neighbors = np.argsort(-1 * np.abs(self.similarity_matrix[int(user)]))
        valid_neighbors = self.baseline_recommender.train[self.baseline_recommender.train['item'] == item][
            'user'].values
        numerator, denominator = 0, 0
        count = 0
        for neighbor in closest_neighbors:
            if count == 3:
                break
            if neighbor != user and neighbor in valid_neighbors:
                count += 1
                numerator += self.similarity_matrix[int(user), int(neighbor)] * \
                             self.baseline_recommender.users_avg_rating.loc[neighbor, item]
                denominator += abs(self.similarity_matrix[int(user), int(neighbor)])

        raw_prediction = self.baseline_recommender.predict(user, item, timestamp) + numerator / denominator
        if raw_prediction > 5:
            return 5
        elif raw_prediction < 0.5:
            return 0.5
        else:
            return raw_prediction

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        return self.similarity_matrix[user1, user2]


def is_weekend(timestamp):
    if datetime.datetime.fromtimestamp(
            timestamp).weekday() == 5 or datetime.datetime.fromtimestamp(timestamp).weekday() == 4:
        return 1
    return 0


def is_day(timestamp):
    if 6 <= datetime.datetime.fromtimestamp(timestamp).hour < 18:
        return 1
    return 0


class LSRecommender(Recommender):
    def __init__(self, ratings: pd.DataFrame):
        self.baseline_recommender = None
        self.number_of_users = None
        self.number_of_items = None
        self.y = None
        self.beta = None
        super().__init__(ratings)

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.baseline_recommender = BaselineRecommender(ratings)
        self.y = self.baseline_recommender.train['rating'].values
        self.number_of_users = self.baseline_recommender.users_avg_rating.shape[0]
        self.number_of_items = self.baseline_recommender.items_avg_rating.shape[0]

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        prediction = self.baseline_recommender.R_hat + self.beta[int(user)] + self.beta[
            int(self.number_of_users - 1 + item)]
        if is_day(timestamp):
            prediction += self.beta[int(self.number_of_items - 1 + self.number_of_users + 1)]
        else:
            prediction += self.beta[int(self.number_of_items - 1 + self.number_of_users + 2)]
        if is_weekend(timestamp):
            prediction += self.beta[int(self.number_of_items - 1 + self.number_of_users + 3)]
        if prediction > 5:
            return 5
        elif prediction < 0.5:
            return 0.5
        else:
            return prediction

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        X = np.zeros((self.baseline_recommender.train.shape[0], self.number_of_items + self.number_of_users + 3))
        for i in range(X.shape[0]):
            X[i, int(self.baseline_recommender.train['user'][i])] = 1
            X[i, int(self.number_of_users - 1 + self.baseline_recommender.train['item'][i])] = 1
            X[i, int(self.number_of_items - 1 + self.number_of_users + 1)] = is_day(
                self.baseline_recommender.train['timestamp'][i])
            X[i, int(self.number_of_items - 1 + self.number_of_users + 2)] = 0 if \
                is_day(self.baseline_recommender.train['timestamp'][i]) == 1 else 0
            X[i, int(self.number_of_items - 1 + self.number_of_users + 3)] = is_weekend(
                self.baseline_recommender.train['timestamp'][i])
        self.beta = np.linalg.lstsq(X, self.y)[0]
        return X, self.beta, self.y


class CompetitionRecommender(Recommender):
    def __init__(self, ratings: pd.DataFrame):
        self.train = None
        self.users_avg_rating = None
        self.items_avg_rating = None
        self.R_hat = None
        self.b_weekend = None
        self.b_weekday = None
        super().__init__(ratings)

    def compute_week_mean(self):
        count_weekend = 0
        count_weekday = 0
        sum_weekend = 0
        sum_weekday = 0
        for row in self.train.index:
            timestamp = self.train['timestamp'][row]
            if is_weekend(timestamp):
                count_weekend += 1
                sum_weekend += self.train['rating'][row]
            else:
                count_weekday += 1
                sum_weekday += self.train['rating'][row]
        return sum_weekday / count_weekday, sum_weekend / count_weekend

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.train = ratings.copy()
        self.R_hat = np.mean(self.train['rating'].values)
        self.train['rating'] = self.train['rating'] - self.R_hat
        self.users_avg_rating = self.train.pivot(index='user', columns='item', values='rating')
        self.users_avg_rating['mean'] = self.users_avg_rating.mean(axis=1)
        self.items_avg_rating = self.train.pivot(index='item', columns='user', values='rating')
        self.items_avg_rating['mean'] = self.items_avg_rating.mean(axis=1)
        self.b_weekday, self.b_weekend = self.compute_week_mean()

    def predict(self, user: int, item: int, timestamp: int, ) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raw_prediction = self.R_hat + 0.6 * self.users_avg_rating.loc[user]['mean'] + \
                         self.items_avg_rating.loc[item]['mean']
        if is_weekend(timestamp):
            raw_prediction += self.b_weekend
        else:
            raw_prediction += self.b_weekday

        if raw_prediction > 5:
            return 5
        elif raw_prediction < 0.5:
            return 0.5
        else:
            return raw_prediction
