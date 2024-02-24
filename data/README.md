### `base.py`

- The utils for loading data, in the form of a base dataloader class, from which other will inherit


### `simulated_data.py`

- As the name suggests, this is a dataloader that generates a full dataset of artificial data, starting with some proportion of them being labelled already
- Because it works with a generative model, it requires access to the non-linear regression and noise models that we are using to fit real data. See relevant subdirectories for those...

