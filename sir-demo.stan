// Deterministic process with stoachastic observations

data {
    int<lower=1> T;          // number of days
    int<lower=1> N;          // total population
    array[T] int<lower=0> cases; // observed new infections per day
}

parameters {
    real<lower=1.01, upper=3> R0;
    real<lower=2, upper=6> D;
}

transformed parameters {
    real gamma = 1 / D;
    real beta = R0 * gamma;
    
    array[T] real<lower=0> S;
    array[T] real<lower=0> I;
    array[T] real<lower=0> R;
    array[T] real lambda_infect;

    
    // Initial conditions
    S[1] = 0.99 * N;
    I[1] = 0.01 * N;
    R[1] = 0;
    
    // Deterministic process
    for (t in 1:(T-1)) {
        lambda_infect[t] = beta * S[t] * I[t] / N;
        real lambda_recover = gamma * I[t];
        
        S[t+1] = S[t] - lambda_infect[t];
        I[t+1] = I[t] + lambda_infect[t] - lambda_recover;
        R[t+1] = R[t] + lambda_recover;
    }
    // Last day lambda
    lambda_infect[T] = beta * S[T] * I[T] / N;
}

model {
    // Priors
    R0 ~ uniform(1.01, 3);
    D ~ uniform(2, 6);
    
    // Stochastic observation model
    for (t in 1:T) {
        cases[t] ~ poisson(lambda_infect[t]);
    }
}