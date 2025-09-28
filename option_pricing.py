import numpy as np

# === Base Class ===
class Option:
    def __init__(self, S0, K, r, T, sigma, option_type="call"):
        """
        Generic Option class
        """
        self.S0 = S0
        self.K = K   
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type

    def payoff(self, S):
        """Compute option payoff at maturity"""
        if self.option_type == "call":
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)


# === American Option ===
class AmericanOption(Option):

    def price_binomial(self, steps=1000):
        """ Cox-Ross-Rubinstein Binomial Model """
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.r * dt) - d) / (u - d)

        # Stock price at maturity
        S = np.array([self.S0 * (u**(steps-j)) * (d**j) for j in range(steps + 1)])
        V = self.payoff(S)

        # Backward induction
        for t in range(steps - 1, -1, -1):
            S = S[0:t+1] / u
            V = np.exp(-self.r * dt) * (q * V[0:t+1] + (1 - q) * V[1:t+2])
            V = np.maximum(V, self.payoff(S))

        return float(V[0].round(4))

    def price_lsm(self, paths=50000, steps=100):
        """ LSM Monte Carlo """
        dt = self.T / steps
        disc = np.exp(-self.r * dt)

        # Simulate stock paths
        S = np.zeros((paths, steps+1))
        S[:, 0] = self.S0
        for t in range(1, steps+1):
            z = np.random.normal(size=paths)
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*z)

        payoff = self.payoff(S)
        cashflow = payoff[:, -1]

        # Backward induction
        for t in range(steps-1, 0, -1):
            itm = payoff[:, t] > 0
            if np.any(itm):
                X = np.vstack([np.ones(np.sum(itm)), S[itm, t], S[itm, t]**2]).T
                y = cashflow[itm] * disc
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                continuation = coeffs[0] + coeffs[1]*S[itm, t] + coeffs[2]*S[itm, t]**2
                exercise = payoff[itm, t] > continuation
                cashflow[itm] = np.where(exercise, payoff[itm, t], cashflow[itm]*disc)
            cashflow[~itm] *= disc

        return float((np.mean(cashflow) * np.exp(-self.r * dt)).round(4))

    # --- Greeks via finite differences ---
    def greeks(self, bump=1e-4, method="binomial"):
        """
        Compute Greeks using finite difference approximations.
        Default pricing method: binomial tree (stable).
        """
        price_func = self.price_binomial if method == "binomial" else self.price_lsm
        base_price = price_func()

        # Delta
        up = AmericanOption(self.S0 + bump, self.K, self.r, self.T, self.sigma, self.option_type)
        down = AmericanOption(self.S0 - bump, self.K, self.r, self.T, self.sigma, self.option_type)
        delta = (up.price_binomial() - down.price_binomial()) / (2*bump)

        # Gamma
        gamma = (up.price_binomial() - 2*base_price + down.price_binomial()) / (bump**2)

        # Vega
        up_sigma = AmericanOption(self.S0, self.K, self.r, self.T, self.sigma + bump, self.option_type)
        vega = (up_sigma.price_binomial() - base_price) / bump

        # Theta (1-day decrease in maturity)
        small_dt = 1/365
        if self.T > small_dt:  # prevent negative maturity
            shorter = AmericanOption(self.S0, self.K, self.r, self.T - small_dt, self.sigma, self.option_type)
            theta = (shorter.price_binomial() - base_price) / (-small_dt)
        else:
            theta = np.nan

        # Rho
        up_r = AmericanOption(self.S0, self.K, self.r + bump, self.T, self.sigma, self.option_type)
        rho = (up_r.price_binomial() - base_price) / bump

        return {
            "Price": round(base_price, 4),
            "Delta": round(delta, 4),
            "Gamma": round(gamma, 4),
            "Vega": round(vega, 4),
            "Theta": round(theta, 4),
            "Rho": round(rho, 4)
        }