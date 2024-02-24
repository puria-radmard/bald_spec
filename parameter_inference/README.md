### `mcmc_base.py`

- Since evaluating the unnormalised posterior pi is a team effort from the dataloader, the hypothesis space prior, the nonlinear regression model, and the noise model, I'm going to make the bold design choice to put it as a class method of the MCMC sampler


### `metropolis_hastings.py`

- We will start with Metropolis Hastings with a normal kernel!

