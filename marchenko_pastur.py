from scipy.stats import rv_continuous
import numpy as np

class marchenko_pastur(rv_continuous):
    def _pdf(self, x, Q, var):
        lambda_min = var * (1 + 1/Q - 2 * np.sqrt(1/Q))
        lambda_max = var * (1 + 1/Q + 2 * np.sqrt(1/Q))
        if x > lambda_max:
            return 0
        if x < lambda_min:
            return 0
        return Q/(2*np.pi*var) * np.sqrt((lambda_max-x)*(x-lambda_min)) / x

    def _cdf(self, x, Q, var):
        lambda_min = var * (1 + 1/Q - 2 * np.sqrt(1/Q))
        lambda_max = var * (1 + 1/Q + 2 * np.sqrt(1/Q))
        lambda_Q = 1/Q
        def r(x):
            return np.sqrt((lambda_max-x) / (x-lambda_min))
        def F(x):
            return 1/(2*np.pi*lambda_Q) * (np.pi*lambda_Q + 1/var * np.sqrt((lambda_max-x)*(x-lambda_min))
                 - (1 + lambda_Q) * np.arctan((r(x)*r(x)-1)/(2 * r(x)))
                 + (1 - lambda_Q) * np.arctan((lambda_min*r(x)*r(x)-lambda_max)/(2 * var * (1-lambda_Q) * r(x))))
        if lambda_Q > 1:
            if x < 0:
                return 0
            elif x < lambda_min:
                return (lambda_Q-1)/(lambda_Q)
            elif x < lambda_max:
                return (lambda_Q-1)/(2 * lambda_Q) + F(x)
            else:
                return 1
        else:
            if x < lambda_min:
                return 0
            if x < lambda_max:
                return F(x)
            else:
                return 1