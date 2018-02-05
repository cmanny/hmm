import numpy as np
from numpy import linalg as la
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# HMM With EM

def norm(a, axis=-1, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

class GaussianHMM(object):

    def __init__(self, n_latent_states):
        self.K = n_latent_states
        self.rand = np.random.RandomState(4)

        self.state_prior = norm(self.rand.rand(self.K))
        self.current_state = self.state_prior
        self.transitions = norm(self.rand.rand(self.K, self.K), axis=1)

        self.state_means = None
        self.state_covs = None
        self.D = None

        self.setup = False

        self.likelihoods = None

    def parameters(self, state_prior, transitions, means, covs):
        self.state_prior = state_prior
        self.transitions = transitions
        self.state_means = means
        self.state_covs = covs
        self.setup = True

    def generate(self):
        while True:
            self.current_state = np.dot(self.transitions, self.current_state.ravel())
            s = np.argmax(np.random.multinomial(1, self.current_state))
            output = np.random.multivariate_normal(
                self.state_means[s],
                self.state_covs[s]
            )
            yield output

    def _setup_emissions(self, data):
        self.D = data.shape[1]
        if not self.setup:
            choices = self.rand.choice(
                np.arange(self.D),
                size=self.K,
                replace=False
            )
            # The selection of means from the original data has a large
            # impact on the success of the algorithm
            self.state_means = data[choices, :]
            self.state_covs = np.zeros((self.K, self.D, self.D))
            self.state_covs += np.eye(self.D)[None, :, :]
        self.setup = True

    def _emissions(self, data):
        emissions = np.zeros((self.K, data.shape[0]))
        for s in range(self.K):
            # For each state, calculate the probability that the data
            # originates from the gaussian associated with it
            # p(x_t | z_t)
            # They can be and usually are larger than 1
            emissions[s, :] = multivariate_normal.pdf(
                data,
                mean=self.state_means[s, :],
                cov=self.state_covs[s, :, :]
            )
        # Transpose to make time the first dimension
        return norm(emissions.T, axis=1)

    def _forward(self, emissions):
        alphas = np.zeros(emissions.shape)
        alphas[0, :] = emissions[0, :] * self.state_prior
        for t in range(1, alphas.shape[0]):
            # In this step we calculate the joint probability
            # alpha(z_n) = p(x_1, ..., x_n, z_n) =
            # p(x_n | z_n) * sum_{z_{n-1}} alpha(z_{n-1}) * p(z_n | z_{n-1})
            # which corresponds to applying the transposed transition matrix
            # to the previous alpha(z_{n-1}) vector, multiplied by the
            # emissions in that time step (since we fix the target state).
            # The likelihood of the sequence is then the sum of log of the
            # final alphas
            alphas[t, :] = norm(
                emissions[t, :] * np.dot(
                    self.transitions.T,
                    alphas[t - 1]
                )
            )
        return alphas, -np.log(np.sum(alphas[-1, :]))

    def _backward(self, emissions):
        betas = np.zeros(emissions.shape)
        betas[-1, :] = np.ones(betas.shape[1])
        for t in range(emissions.shape[0] - 1)[::-1]:
            # In this step we calculate the conditional probability
            # beta(z_n) = p(x_{n+1}, ..., x_N | z_n) =
            # sum_{z_{n+1}} beta(z_{n+1}) * p(x_{n+1} | z_{n+1}) * p(z_{n+1} | z_n)
            # which corresponds to applying the transition matrix to the
            # previous beta and emissions (since we fix the source state)
            betas[t, :] = norm(
                np.dot(
                    self.transitions,
                    betas[t + 1, :] * emissions[t + 1, :]
                )
            )
        return betas

    def _em_step(self, data):
        T = data.shape[0]
        emissions = self._emissions(data)
        alphas, log_likelihood = self._forward(emissions)
        betas = self._backward(emissions)

        epsilon = np.zeros((self.K, self.K))
        # Using alpha and beta, we calculate the expectation in state transitions
        # and normalise across rows to create a valid transition matrix
        for t in range(T - 1):
            epsilon += norm(
                self.transitions * np.dot(
                    alphas[t, :],
                    (betas[t + 1, :] * emissions[t + 1, :]).T
                )
            )
        gamma = norm(alphas * betas, axis=1)
        gamma_state_sum = np.sum(gamma, axis=0)
        gamma_state_sum[gamma_state_sum < 1e-32] = 1

        self.transitions = norm(epsilon, axis=1)
        self.state_prior = gamma[0, :]
        for i in range(self.K):
            gamma_data = data.T * gamma[:, i]
            self.state_means[i, :] = np.sum(gamma_data, axis=1) / gamma_state_sum[i]
            for t in range(T):
                d = data[t, :] - self.state_means[i, :]
                self.state_covs[i, :, :] += gamma[t, i] * np.outer(d, d)
            self.state_covs[i, :, :] /= gamma_state_sum[i]
        return log_likelihood

    def log_likelihood(self, data):
        return self._forward(self._emissions(data))[1]

    def inference(self, data):
        self._setup_emissions(data)
        for _ in range(15):
            print("NLL: {}".format(self._em_step(data)))
        return self.log_likelihood(data)

    def info(self):
        print("Transitions:")
        print("{}".format(str(self.transitions)))

        print("Parameters:")
        for s in range(self.K):
            print("state {}".format(s))
            print("mean: {}".format(str(self.state_means[s])))
            print("diag(cov): {}".format(str(np.diag(self.state_covs[s]))))



def generate_rand_sequence(rand_state, n_features, count):
    return norm(rand_state.rand(count, n_features), axis=1)

def random_test():
    rand_state = np.random.RandomState(0)
    dimensions = 8
    sequence_length = 40
    data = generate_rand_sequence(rand_state, 2, 30)
    test = generate_rand_sequence(rand_state, 2, 30)

    model = GaussianHMM(2)
    model.inference(data)

    model_likelihood = model.log_likelihood(data)
    test_likelihood = model.log_likelihood(test)
    print("Final Likelihood {}".format(model_likelihood))
    print("Test Likelihood {}".format(test_likelihood))

def recreate_test():
    state_prior = norm(np.ones(4))
    state_means = np.array([
        1*np.zeros(5),
        -1*np.ones(5),
        5*np.ones(5),
        -5*np.ones(5)
    ])
    state_covs = np.array([np.eye(5) for _ in range(4)])
    transitions = norm(np.ones((4, 4)), axis=1)
    model = GaussianHMM(4)
    model.parameters(state_prior, transitions, state_means, state_covs)
    samples = []
    for i in range(200):
        sample = model.generate().next()
        samples.append(sample)
    recreated_model = GaussianHMM(4)
    recreated_model.inference(np.array(samples))
    recreated_model.info()
    for i in range(10):
        sample = recreated_model.generate().next()
        print(sample)



if __name__ == "__main__":
    recreate_test()
